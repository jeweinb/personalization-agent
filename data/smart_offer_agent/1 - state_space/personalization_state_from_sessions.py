# Databricks notebook source
spark.sql('set spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;')
spark.sql('set spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;')
spark.sql('set spark.sql.sources.partitionOverwriteMode = dynamic;')

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pyspark.sql.types as T

import os
import yaml
from yaml import full_load

with open('../../../agent/app_configs/conf1.yaml', 'r') as f:
    configs = yaml.full_load(f)

MAX_STATE_DIM = configs['state_max_len']

# COMMAND ----------

sessions = (spark.table('prod_oco_team_data_science.consumer_store.consumer_sessions')
            .filter('''event_type <> 'text' and date >= date_sub(current_date, 180)''')
            ).cache()

# COMMAND ----------

state_window = (Window.partitionBy('enterprise_id')
                .orderBy(F.col('date'))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing))

state = (
    sessions
    .select(
        'enterprise_id',
        'date',
        'events', 
        'event_tokens'
    )
    .withColumn('state', F.flatten(F.collect_list('events').over(state_window)))
    .withColumn('state_tokenized', F.flatten(F.collect_list('event_tokens').over(state_window)))
    .drop('date', 'events', 'event_tokens')
    .distinct()
    .withColumn('orig_size', F.size(F.col('state')))
    .withColumn('state', F.when(F.col('orig_size') > MAX_STATE_DIM,
                                F.slice('state', F.col('orig_size') - F.lit(MAX_STATE_DIM),
                                        F.col('orig_size')))
                .otherwise(F.col('state')))
    .withColumn('state_tokenized', F.when(F.col('orig_size') > MAX_STATE_DIM,
                                          F.slice('state_tokenized', F.col('orig_size') - 
                                                  F.lit(MAX_STATE_DIM), F.col('orig_size')))
                .otherwise(F.col('state_tokenized')))
)

# COMMAND ----------

(
    state
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(f"prod_oco_team_data_science.personalization_agent.personalization_state")
)
