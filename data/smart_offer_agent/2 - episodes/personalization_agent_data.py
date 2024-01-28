# Databricks notebook source
import os, json, time, random, yaml
from datetime import datetime
import pytz
from pytz import timezone
import pandas as pd
import numpy as np
from itertools import chain
from functools import partial

import torch
from torch.nn.functional import softmax
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import udf
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import VectorAssembler
import pyspark.ml.functions as mlF
from pyspark.ml.pipeline import Pipeline

from agent.state_encoder.model import AttentionEncoderDecoder


eid_config = {
    'ivr': 'cagm',
    'spec-pat': 'cagm',
    'advocacy': 'enterprise_id',
}
mapping_expr = F.create_map([F.lit(x) for x in chain(*eid_config.items())])

def get_modified_time(p):
    unaware_est = datetime.fromtimestamp(os.path.getmtime(p))
    localtz = timezone('UTC')
    aware_est = unaware_est.replace(tzinfo=localtz)
    return aware_est

def make_probs(logits, log=False):
    logits = [float(i) for i in logits]
    logits = [max(logits) if l == -1000 else l for l in logits]
    probs = np.array(softmax(torch.tensor(logits), dim=0))
    if log:
        probs = np.log(probs)
    return probs.tolist()

def re_encode_state(dat, ckpt):
    encoder = AttentionEncoderDecoder.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))
    encoder.eval()
    
    def f(s):
        if not isinstance(s, (np.ndarray, list)):
            s = [0]
        tokens = torch.tensor(s).long()
        obs = encoder.encode(s).squeeze().tolist()
        return obs
    return dat.apply(f)

schema = T.ArrayType(T.DoubleType())
encoder_udf = pandas_udf(re_encode_state, returnType=schema, functionType=F.PandasUDFType.SCALAR)

make_probs_udf = F.udf(partial(make_probs, log=False), T.ArrayType(T.DoubleType()))
make_logs_udf = F.udf(partial(make_probs, log=True), T.ArrayType(T.DoubleType()))

# COMMAND ----------


#TODO the enterprise_id field was not implemented until ~09/01/2023
# go back and pre-processing the agent_data_raw table is clean up all the changes that have been incrementally made over the last year while preserving as much historical data as possible.
raw_data = spark.table('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw').filter('enterprise_id is not null')
state_data = spark.table('prod_oco_team_data_science.personalization_agent.personalization_state')

rewards = spark.table('prod_oco_team_data_science.personalization_agent.personalization_reward')
dispositions = spark.table('prod_oco_team_data_science.personalization_agent.personalization_offer_dispositions')
dispositions = dispositions.groupBy('enterprise_id', 'date', 'source').agg(F.sum(F.col('accepted')).alias('rewards'))
rewards = rewards.groupBy('enterprise_id', 'date', 'source').agg(F.sum('reward_value').alias('rewards'))
rew_dispo = dispositions.union(rewards).groupBy('enterprise_id', 'date', 'source').agg(F.sum('rewards').alias('rewards'))

rewards = (rewards
           .withColumnRenamed('source', 'implementation')
           .withColumn('time', F.to_timestamp('date'))
           .select(
               'implementation',
               'enterprise_id',
               'date',
               'time',
               F.lit(F.array(F.lit(0))).alias('actions'),
               F.lit(None).alias('action_prob'),
               F.lit(None).alias('action_logp'),
               'rewards',
           )
)

episodes = raw_data.select(
    'implementation',
    'enterprise_id',
    F.to_date('time').alias('date'),
    'time',
    'actions',
    'action_prob',
    'action_logp',
    'rewards'
)

episodes = episodes.union(rewards)
episodes = (episodes
            .join(state_data.select('enterprise_id', 'date', 'state_tokenized'),
                  on=['enterprise_id', 'date'])
            .withColumn('obs', encoder_udf(F.col("state_tokenized")))
            .drop('state_tokenized')
)

episodes.display()

# COMMAND ----------


print('valid agent data rows: ', episodes.count())
ep_win = Window().partitionBy('enterprise_id').orderBy('time')
episodes = episodes.withColumn('t', F.row_number().over(ep_win) - F.lit(1))
episodes = episodes.withColumnRenamed('best_action', 'actions')
episodes = episodes.withColumn('call_penalty', F.coalesce('call_penalty', F.lit(0)))
episodes = episodes.withColumn('rewards', F.col('accepted_reward') + F.col('call_penalty')).drop('accepted_reward', 'call_penalty')
episodes = episodes.withColumn('new_obs', F.lead('obs').over(ep_win))
episodes = episodes.withColumn('new_updated_obs', F.lead('updated_obs').over(ep_win))
episodes = episodes.withColumn('last_obs', F.last('obs', ignorenulls=True).over(ep_win))
episodes = episodes.withColumn('new_obs', F.coalesce('new_obs', 'last_obs')).drop('last_obs')
episodes = episodes.withColumn('last_updated_obs', F.last('updated_obs', ignorenulls=True).over(ep_win))
episodes = episodes.withColumn('new_updated_obs', F.coalesce('new_updated_obs', 'last_updated_obs')).drop('last_updated_obs')

episodes = episodes.withColumn('new_mask', F.lead('mask').over(ep_win))
episodes = episodes.withColumn('last_mask', F.last('mask', ignorenulls=True).over(ep_win))
episodes = episodes.withColumn('new_mask', F.coalesce('new_mask', 'last_mask')).drop('last_mask')
done_win = Window().orderBy('cagm', 'time')
episodes = episodes.withColumn('next_cagm', F.lead('cagm').over(done_win))
episodes = episodes.withColumn('dones', F.when(F.col('cagm') != F.col('next_cagm'), 1).otherwise(0)).drop('next_cagm')
episodes = episodes.withColumn('action_prob', make_probs_udf('logits')[F.col('actions')])
episodes = episodes.withColumn('action_logp', make_logs_udf('logits')[F.col('actions')]).drop('logits')
episodes = episodes.withColumn('q_value', F.col('q_values')[F.col('actions')])

episodes = episodes.fillna(0, subset=['rewards'])

min_len = episodes.filter(f't >= {MIN_EPISODE_LEN - 1}').select('cagm').distinct()
episodes = episodes.join(min_len, on='cagm').orderBy('cagm', 'time')
print('valid agent data rows: ', episodes.count())
# print(f'episodes with {MIN_EPISODE_LEN} or more steps: ', episodes.select('cagm').distinct().count())

EPS_ID_MAP = {k: i for i, k in enumerate(episodes.select('cagm').distinct().toPandas()['cagm'].tolist())}
eps_id_mapper = F.create_map([F.lit(x) for x in chain(*EPS_ID_MAP.items())])
episodes = episodes.withColumn('eps_id', eps_id_mapper[F.col('cagm')]).cache()

# print('final agent data rows: ', episodes.count())
(episodes.write
 .mode("overwrite")
 .option('overwriteSchema', 'true')
 .saveAsTable(f"{target_schema}.personalization_training_episodes", path=f'{target_mount}/personalization/personalization_training_episodes_table')
)

(spark.table(f"{target_schema}.personalization_training_episodes")
 .coalesce(1)
 .write
 .mode("overwrite")
 .parquet('/mnt/gold/personalization/personalization_training_episodes')
)

# COMMAND ----------

call_obviation = spark.table('prod_oco_team_data_science.silver.xxxrx_call_log_topic')
agent_raw_data = (spark.table('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw')
                  .filter(f'''app in ('ivr', 'spec-pat', 'advocacy') and time >= '2022-12-01' ''')
                  .withColumn('id_type', mapping_expr[F.col('app')])
                  .withColumnRenamed('enterprise_id', 'id')
                  .withColumn('id', F.coalesce('cagm', 'id'))
)
eid_map = (spark.table('prod_oco_team_data_science.silver.eid_to_alternative_id')
           .dropDuplicates()
           .withColumnRenamed('alternative_id', 'id')
)
agent_raw_data = agent_raw_data.join(eid_map, on=['id_type', 'id'])

# agent_raw_data = call_obviation.

# COMMAND ----------

agent_ts = agent_data.groupby('date', 'tfn').agg(F.count('*').alias('agent_interactions')).orderBy('date')
ivr = ivr.withColumn('date', F.to_date('startdtm'))
ivr = ivr.filter('tfn = 8778896510 and date between "2022-09-30" and current_date() and callertype="member"')
ivr_ts = ivr.groupby('date', 'tfn').agg(F.count('*').alias('ivr_calls')).orderBy('date')
ep_ts = episodes.withColumn('date', F.to_date('time'))
ep_ts = ep_ts.groupby('date', 'tfn').agg(F.count('*').alias('valid_interactions')).orderBy('date')
agent_ivr_ts = agent_ts.join(ivr_ts, on=['date', 'tfn']).join(ep_ts, on=['date', 'tfn'])

no_interactions = ivr.join(agent_data.select('CallGUID').distinct(), on='CallGUID', how='leftanti')
agent_ivr_ts.select('date', 'tfn', 'ivr_calls', 'agent_interactions', 'valid_interactions').orderBy('date').display()
