# Databricks notebook source
dbutils.widgets.text('start_date', '2022-10-01')
dbutils.widgets.text('toolbox_path', '/Workspace/Repos/prod_repos/oco-ds-toolbox/src')
dbutils.widgets.text('output_db', 'prod_oco_team_data_science.personalization_agent')
dbutils.widgets.text('table_name', 'personalization_reward')

start_date = dbutils.widgets.get('start_date')
output_db = dbutils.widgets.get('output_db')
table_name = dbutils.widgets.get('table_name')

# COMMAND ----------

import sys
sys.path.append(dbutils.widgets.get('toolbox_path'))

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row
from pyspark.sql import Window
from delta.tables import DeltaTable

from oco_ds.toolbox.utils.spark.delta import create_table_if_not_exists

# COMMAND ----------

schema = T.StructType([
    T.StructField('enterprise_id', T.LongType()),
    T.StructField('organization', T.StringType()),
    T.StructField('app', T.StringType()),
    T.StructField('source', T.StringType()),
    T.StructField('category', T.StringType()),
    T.StructField('date', T.DateType()),
    T.StructField('date_time', T.TimestampType()),
    T.StructField('reward_name', T.StringType()),
    T.StructField('reward_desc', T.StringType()),
    T.StructField('reward_value', T.IntegerType()),
])

create_table_if_not_exists(f"{output_db}.{table_name}", schema, ['date'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rx repeat call penalty (call obviation)

# COMMAND ----------

CALL_WINDOW = 7
REPEAT_CALL_PENALTY = -3

call = spark.sql(f"""
          select call_guid, start_dtm as date_time, cagm
          from prod_oco_team_data_science.gold.call_log_topic
          where logical_app_name = 'RX'
          and caller_type = 'member'
          and start_dtm > '{start_date}'
""")

eid = (spark.table('prod_oco_team_data_science.gold.eid_to_alternative_id')
       .filter('''id_type = 'cagm' ''')
       .select('enterprise_id', F.col('alternative_id').alias('cagm'))
)
call = call.join(eid, on='cagm').drop('cagm')

call.createOrReplaceTempView("calltmp")
call = spark.sql(f"""SELECT enterprise_id, date_time,
                    min({REPEAT_CALL_PENALTY}) OVER (
                    PARTITION BY enterprise_id
                    ORDER BY cast(date_time as timestamp)
                    RANGE BETWEEN INTERVAL {CALL_WINDOW} DAYS PRECEDING AND INTERVAL 1 DAYS PRECEDING) as reward_value
                    from calltmp""").filter('reward_value is not null')

output_table = call.select('enterprise_id',
                   F.lit('xxx').alias('organization'),
                   F.lit('ivr').alias('app'),
                   F.lit('xxxrx').alias('source'),
                   F.lit('call').alias('category'),
                   F.to_date('date_time').alias('date'),
                   'date_time',
                   F.lit('repeat_call').alias('reward_name'),
                   F.lit(f'repeat call within {CALL_WINDOW} days').alias('reward_desc'),
                   'reward_value').distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carepathways Stickiness and Gap Closure
# MAGIC Use evisor rules to get gap closure. Use repeat visits to CPW as stickiness.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving final table

# COMMAND ----------

(
    output_table
    .repartition('date')
    .sortWithinPartitions('date', 'enterprise_id')
    .write
    .insertInto(f"{output_db}.{table_name}", overwrite=True)
)

# COMMAND ----------

DeltaTable.forName(spark, f"{output_db}.{table_name}").vacuum()
