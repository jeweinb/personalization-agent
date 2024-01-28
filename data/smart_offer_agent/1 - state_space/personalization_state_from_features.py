# Databricks notebook source
# spark.sql('set spark.databricks.delta.autoCompact.enabled=auto')
# spark.sql('set spark.databricks.delta.optimizeWrite.enabled=true')
# spark.sql('set spark.sql.sources.partitionOverwriteMode=dynamic;')
spark.sql('set spark.databricks.delta.retentionDurationCheck.enabled=false;')

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pyspark.sql.types as T
import tempfile

def clear_computation_graph(data_frame):
   """Returns a ‘cleared’ dataframe after saving it for PySpark to work from 
    
    This will 'clear' the computation graph for you
    since occasionally spark will poorly optimize commands.
   This is useful to avoid having too many nested operations 
    especially more complex ones that tank performance.
   Keyword arguments:
   data_frame -- the dataframe you want to clear the graph for
   spark_session -- your current spark session
   """
   with tempfile.TemporaryDirectory() as path:
       data_frame.write.parquet(path, mode="overwrite")
       data_frame = spark.read.parquet(path)
       return data_frame

import os
import yaml
from yaml import full_load

with open('../../../agent/app_configs/conf1.yaml', 'r') as f:
    configs = yaml.full_load(f)

MAX_STATE_DIM = int(configs['state_max_len'])
TOKEN_PATH = 'prod_oco_team_data_science.personalization_agent.personalization_tokens'

# COMMAND ----------

def tokenizer(df, token_col, table_path):
    token_win = Window.orderBy(token_col)
    token_df = df.select(token_col).distinct()

    save_table_exists = spark._jsparkSession.catalog().tableExists(table_path)
    if save_table_exists:
        existing_tokens = spark.table(table_path)
        max_token = (existing_tokens.select(F.max('token').alias('token'))
                     .pandas_api()['token'].tolist()[0])
        existing_tokens = existing_tokens.drop('token')

        new_tokens = token_df.join(existing_tokens, on=token_col, how='left_anti')
        new_tokens = new_tokens.withColumn('token', F.lit(max_token) + F.row_number().over(token_win))
    
        (new_tokens
        .write
        .format('delta')
        .mode('append')
        .option('overwriteSchema', 'true')
        .saveAsTable(f"prod_oco_team_data_science.personalization_agent.personalization_tokens"))
    else:
        token_df = token_df.withColumn('token', F.row_number().over(token_win))

        (token_df
        .write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f"prod_oco_team_data_science.personalization_agent.personalization_tokens"))

# COMMAND ----------

features = (spark.table('prod_oco_team_data_science.consumer_store.consumer_features')
            .filter(f'''(feature_type in ('session_event')
                    or category = 'demographics')
                    and date >= date_sub(current_date, 365)
                    ''')
            .withColumn('token_name', F.concat_ws('___', 'feature_name', 'feature_value'))
)

tokenizer(features, 'token_name', TOKEN_PATH)
tokens = spark.table(TOKEN_PATH)
tokens.coalesce(1).write.parquet('/mnt/ocep/eureka-gold/personalization/personalization_state_train_tokens', mode='overwrite')

# COMMAND ----------

# for every feature/timestamp get last previous N features.
state_window = (Window.partitionBy('enterprise_id')
                .orderBy('date')
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
# for every day in the dataset capture features as they stand end of each day.
size_win = Window.partitionBy('enterprise_id').orderBy(F.col('date').desc())

state_features = (
    features
    .join(tokens, on='token_name')
    .select(
        'enterprise_id',
        'date',
        'date_time',
        'token_name', 
        'token',
    )
    .groupby('enterprise_id', 'date')
    .agg(
        F.sort_array(F.collect_list(F.struct(
            F.concat_ws('-', 'date_time', 'token_name'),
            'token_name'))).alias('state'),
        F.sort_array(F.collect_list(F.struct(
            F.concat_ws('-', 'date_time', 'token_name'),
            'token'))).alias('state_tokenized'),
    )
    .withColumn('state', F.col("state.token_name"))
    .withColumn('state_tokenized', F.col("state_tokenized.token"))
    .withColumn('size', F.size('state'))
)

# state_features = clear_computation_graph(state_features)

state_features = (state_features
                .withColumn('state', F.slice(
                    F.flatten(F.collect_list('state').over(state_window)), 1, MAX_STATE_DIM))
                .withColumn('state_tokenized', F.slice(
                    F.flatten(F.collect_list('state_tokenized').over(state_window)), 1, MAX_STATE_DIM))
)

# state_features = clear_computation_graph(state_features)

current_ind = Window.partitionBy('enterprise_id').orderBy(F.col('date').desc())
output_table = (state_features.withColumn('current_record', F.row_number().over(current_ind))
                .withColumn('current_record', F.when(F.col('current_record') == F.lit(1),
                                                    F.lit(1)).otherwise(F.lit(0))))

# COMMAND ----------

(
    output_table
    .repartition(200)
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(f"prod_oco_team_data_science.personalization_agent.personalization_state")
)

# COMMAND ----------

state = spark.table('prod_oco_team_data_science.personalization_agent.personalization_state')
state.filter('current_record = 1').sample(0.3).repartition(1).write.parquet('/mnt/ocep/eureka-gold/personalization/personalization_state_train', mode='overwrite')

# COMMAND ----------

cagm = (spark.table('prod_oco_team_data_science.gold.eid_to_alternative_id')
              .filter('''id_type = 'cagm' ''')
              .select('enterprise_id', F.col('alternative_id').alias('cagm'))
              .distinct()
              )
cagm_state = state.filter('current_record = 1').join(cagm, on='enterprise_id').cache()
cagm_state = cagm_state.filter('cagm is not null and date >= current_date() - interval 365 day')

(
    cagm_state
    .repartition(20)
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(f"prod_oco_team_data_science.personalization_agent.personalization_state_rx")
)

# COMMAND ----------

advocacy_clients = (spark.table('prod_oco_team_data_science.care_pathways.evisor')
                    .withColumnRenamed('indv_id', 'enterprise_id')
                    .select('enterprise_id')
                    .distinct()
                    )
adv_state = state.filter('current_record = 1').join(advocacy_clients, on='enterprise_id').cache()

(
    adv_state
    .repartition(20)
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(f"prod_oco_team_data_science.personalization_agent.personalization_state_advocacy")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC vacuum prod_oco_team_data_science.personalization_agent.personalization_state retain 24 hours;
# MAGIC vacuum prod_oco_team_data_science.personalization_agent.personalization_state_rx retain 24 hours;
# MAGIC vacuum prod_oco_team_data_science.personalization_agent.personalization_state_advocacy retain 24 hours;

# COMMAND ----------

# MAGIC %sql
# MAGIC optimize prod_oco_team_data_science.personalization_agent.personalization_state zorder by (enterprise_id);
# MAGIC optimize prod_oco_team_data_science.personalization_agent.personalization_state_rx zorder by (cagm);
# MAGIC optimize prod_oco_team_data_science.personalization_agent.personalization_state_advocacy zorder by (enterprise_id);

# COMMAND ----------


