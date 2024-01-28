# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

raw = spark.table('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw')
raw = raw.withColumn('date', F.to_date('time'))
daily_vol = raw.groupby('app', 'date').agg(F.count('*').alias('volume')).orderBy('app', F.col('date').desc())

daily_vol.display()

# COMMAND ----------


