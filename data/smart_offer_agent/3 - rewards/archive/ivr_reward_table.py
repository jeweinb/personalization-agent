# Databricks notebook source
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row
from pyspark.sql import Window

# COMMAND ----------

target_schema = 'prod_oco_team_data_science.adhoc'
target_mount = '/mnt/eureka-gold/adhoc'

# COMMAND ----------

call = spark.sql("""
  select distinct callguid, cagm, StartDTM as starttime, DATE(StartDTM) as startdate, tfn, subfunctiondetail, subfunction, function, finalnltag, xferstatus
  from prod_oco_team_data_science.silver.ivr_xxx_dbo_call_log_topic
  where cagm <> '0000000000|0000000000000000||'
  and authenticated = 1
  and logicalAppName='RX' and callertype='member' """)

peg = spark.table(f'{target_schema}.personalization_smart_offering_metrics')

peg = peg.drop('sms_success')
peg = peg.withColumnRenamed('ordersms_offered', 'sms_offered')
peg = peg.withColumnRenamed('ordersms_success', 'sms_success')
peg = peg.withColumnRenamed('ordersms_accepted', 'sms_accepted')

# COMMAND ----------

metric_cols = [c for c in peg.columns if c not in ['CallGUID', 'startdate']]
metric_array = [F.array(F.lit(c), F.col(c)) for c in metric_cols]

reward_long = (
    peg.withColumn('array_map', F.explode(F.array(*metric_array)))
    .withColumn('metric', F.col('array_map')[0])
    .withColumn('type', F.split('metric', '\_').getItem(1))
    .withColumn('metric', F.split('metric', '\_').getItem(0))
    .withColumn('value', F.col('array_map')[1].cast(T.IntegerType()))
).drop(*metric_cols + ['array_map'])

reward_pivot = reward_long.groupby('CallGUID', 'metric').pivot('type').max('value').fillna(0)
reward_pivot = reward_pivot.join(call, on=['CallGUID'])
reward_pivot = reward_pivot.withColumn('offered_not_accepted', F.when((F.col('offered') == 1) & (F.col('accepted') == 0) & (F.col('success') == 0), 1).otherwise(0))

# COMMAND ----------

reward_table = reward_pivot.withColumn('yearmonth', F.concat_ws('-',F.year('startdate'), F.month('startdate')))
reward_table = reward_table.groupby('metric', 'startdate', 'tfn').agg(F.approx_count_distinct('cagm').alias('members'),
                                                  F.count('*').alias('sessions'),
                                                  F.sum('offered').alias('offered'),
                                                  F.sum('accepted').alias('accepted'),
                                                  F.sum('success').alias('success'),
                                                  F.sum('failure').alias('failure'),
                                                 F.sum('offered_not_accepted').alias('offered_not_accepted'))
reward_table = reward_table.withColumn('member_offer_ratio', F.col('members') / F.col('offered'))
reward_table = reward_table.withColumn('offered_rate', F.col('offered') / F.col('sessions'))
reward_table = reward_table.withColumn('accepted_rate', F.col('accepted') / F.col('offered'))
reward_table = reward_table.withColumn('success_rate', F.col('success') / F.col('offered'))
reward_table = reward_table.orderBy('metric')

reward_table.repartition(1).write.option('overwriteSchema', 'true').\
saveAsTable(f"{target_schema}.personalization_reward_report", mode='overwrite')

# COMMAND ----------

reward_session = reward_pivot.groupby('CallGUID', 'cagm', 'starttime').agg(
                                                  F.sum('offered').alias('offered'),
                                                  F.sum('accepted').alias('accepted'),
                                                  F.sum('success').alias('success'),
                                                  F.sum('failure').alias('failure'),
                                                  F.sum('offered_not_accepted').alias('offered_not_accepted'),
                                                  F.collect_list(F.when(F.col('offered')==1,F.col('metric'))).alias('offered_list'),
                                                  F.collect_list(F.when(F.col('accepted')==1,F.col('metric'))).alias('accepted_list'))

## only consider calls that were transfered to an agent
reward_calls = call.filter('xferstatus = "XFER"').select('cagm', F.col('CallGUID').alias('future_gvpessionid'), F.col('starttime').alias('future_starttime'))
reward_calls = reward_session.select('cagm', 'CallGUID', 'starttime').join(reward_calls, on='cagm').filter('future_starttime > starttime and future_starttime <= starttime + interval 7 days')
reward_calls = reward_calls.groupby('cagm', 'CallGUID', 'starttime').agg(F.count('future_gvpessionid').alias('calls'))

reward_session = reward_session.join(reward_calls, on=['cagm', 'CallGUID', 'starttime'], how='left').fillna(0, subset='calls')
state = spark.table(f'{target_schema}.personalization_cagm_state').select('cagm', F.col('state_tokenized').alias('next_state'))
reward_session = reward_session.join(state, on='cagm')

reward_session = reward_session.withColumn('success_reward', F.col('success') + (F.col('calls') * F.lit(-1)) + (F.col('offered_not_accepted') * F.lit(-1)))
reward_session = reward_session.withColumn('accepted_reward', F.col('accepted') + (F.col('calls') * F.lit(-1)) + (F.col('offered_not_accepted') * F.lit(-1)))

reward_session.display()

# COMMAND ----------

#random sample of members
reward_session.select('cagm', 'CallGUID').distinct().orderBy(F.rand()).limit(30).toPandas().to_dict('records')

# COMMAND ----------

(reward_session.repartition(1).write
 .mode('overwrite')
 .option('overwriteSchema', 'true')
 .saveAsTable(f"{target_schema}.personalization_reward", path=f'{target_mount}/personalization/personalization_reward_table')
)

(spark.table(f"{target_schema}.personalization_reward").coalesce(1)
 .write
 .mode('overwrite')
 .parquet(f'{target_mount}/personalization/personalization_reward')
)
