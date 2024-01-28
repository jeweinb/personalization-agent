# Databricks notebook source
dbutils.widgets.text('start_date', '2022-10-01')
dbutils.widgets.text('toolbox_path', '/Workspace/Repos/prod_repos/oco-ds-toolbox/src')
dbutils.widgets.text('output_db', 'prod_oco_team_data_science.personalization_agent')
dbutils.widgets.text('table_name', 'personalization_offer_dispositions')

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
    T.StructField('offer', T.StringType()),
    T.StructField('offered', T.IntegerType()),
    T.StructField('accepted', T.IntegerType()),
    T.StructField('success', T.IntegerType()),
    T.StructField('failure', T.IntegerType()),
])

create_table_if_not_exists(f"{output_db}.{table_name}", schema, ['date'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rx IVR Dispositions
# MAGIC getting the dispositions directly from the peg log data

# COMMAND ----------

peg = spark.sql(f'''
  select call_guid,
  -- refill 
  max(rx_refilled) as refill_success,
  max(rx_refill_failed) as refill_failure,
  max(refill_alert) as refill_offered, 
  max(want_refill_yes) as WantRefill_Yes, 
   -- schedulerefill
  max(refill_scheduled) as schedulerefill_success,
  max(refill_schedule_failed) as schedulerefill_failure,
  max(refill_schedule_alert) as schedulerefill_offered,
  max(refill_schedule_alert_yes) as schedulerefill_accepted,
  -- renewal
  max(rx_renewed) as renewal_success,
  max(rx_renew_failed) as renewal_failure,
  max(renewal_alert) as renewal_offered,
  max(renewal_alert_yes) as renewal_accepted,
  -- autorefill
  max(auto_refill_enrolled) as autorefill_success,
  max(auto_refill_failed) as autorefill_failure,
  max(auto_refill_alert) as autorefill_offered,
  max(auto_refill_alert_enroll_now) as autorefill_accepted,
  -- outstandingbalance
  max(balance_paid) as outstandingbalance_success,
  max(ob_failed) as outstandingbalance_failure,
  max(case when ob_alert_after_ss = 1 then 1 when peg='56070' and request_content like '%RX_OB_After_SS1%' then 1 else 0 end) as outstandingbalance_offered,
  max(ob_pay_now_alert_yes) as OBPayNowAlert_Yes, 
  -- address
  max(address_release_failed) as address_failure,
  max(address_released) as address_success,
  max(address_alert) as address_offered,
  max(want_fix_address_yes) as WantFixAddress_Yes,
  -- creditcard
  max(case when cc_released=1 or add_new_cc=1 then 1 else 0 end) as creditcard_success,
  max(case when cc_release_failed=1 or add_new_cc_failed=1 then 1 else 0 end) as creditcard_failure,
  max(cc_alert) as creditcard_offered,
  max(want_fix_cc_yes) as WantFixCC_Yes,
  -- sms pharmacy order
  max(pharmacy_order_sms_offered) as ordersms_offered,
  max(case when pharmacy_order_sms_offered_response='yes' then 1 else 0 end) as ordersms_accepted,
  max(case when sms_enrolled=1 or gan_activation_sent=1 then 1 else 0 end) as sms_success,
  max(sms_add_failure) as sms_failure,
  -- sms pharmacy care
  max(pharmacy_care_sms_offered) as pasms_offered,
  max(case when pharmacy_care_sms_offered_response='yes' then 1 else 0 end) as pasms_accepted,
  -- clinicalopportunity
  max(clinical_oppty_alert) as clinicalopportunity_offered,
  max(clinical_oppty_alert_yes) as clinicalopportunity_accepted,
  -- diabeteshold #TODO ask sinan about accepted/success fields
  max(diabetes_hold_alert) as diabeteshold_offered,  
  -- orderstatus
  max(order_status_alert) as orderstatus_offered,
  max(order_status_alert_yes) as orderstatus_accepted
  
  from prod_oco_team_data_science.gold.xxxrx_peg_log_topic
  where app='optOmniMain_xxxRX'
        and start_dtm >= '{start_date}'
  group by call_guid
''').cache()

peg = peg.withColumn('refill_accepted', F.col('refill_offered') * F.col('WantRefill_Yes'))
peg = peg.withColumn('address_accepted', F.col('address_offered') * F.col('WantFixAddress_Yes'))
peg = peg.withColumn('outstandingbalance_accepted', F.col('outstandingbalance_offered') * F.col('OBPayNowAlert_Yes'))
peg = peg.withColumn('creditcard_accepted', F.col('creditcard_offered') * F.col('WantFixCC_Yes'))

peg = peg.withColumn('refill_success', F.col('refill_accepted') * F.col('refill_success'))
peg = peg.withColumn('schedulerefill_success', F.col('schedulerefill_accepted') * F.col('schedulerefill_success'))
peg = peg.withColumn('renewal_success', F.col('renewal_accepted') * F.col('renewal_success'))
peg = peg.withColumn('autorefill_success', F.col('autorefill_accepted') * F.col('autorefill_success'))
peg = peg.withColumn('outstandingbalance_success', F.col('outstandingbalance_accepted') * F.col('outstandingbalance_success'))
peg = peg.withColumn('address_success', F.col('address_accepted') * F.col('address_success'))
peg = peg.withColumn('creditcard_success', F.col('creditcard_accepted') * F.col('creditcard_success'))
peg = peg.withColumn('ordersms_success', F.col('ordersms_accepted') * F.col('sms_success'))
peg = peg.withColumn('pasms_success', F.col('pasms_accepted') * F.col('sms_success'))
peg = peg.drop('WantRefill_Yes', 'WantFixAddress_Yes', 'OBPayNowAlert_Yes', 'WantFixCC_Yes')

peg = peg.drop('sms_success')
peg = peg.withColumnRenamed('ordersms_offered', 'sms_offered')
peg = peg.withColumnRenamed('ordersms_success', 'sms_success')
peg = peg.withColumnRenamed('ordersms_accepted', 'sms_accepted')

# COMMAND ----------

metric_cols = [c for c in peg.columns if c not in ['call_guid']]
metric_array = [F.array(F.lit(c), F.col(c)) for c in metric_cols]

disposition_long = (
    peg.withColumn('array_map', F.explode(F.array(*metric_array)))
    .withColumn('offer', F.col('array_map')[0])
    .withColumn('disposition', F.split('offer', '\_').getItem(1))
    .withColumn('offer', F.split('offer', '\_').getItem(0))
    .withColumn('value', F.col('array_map')[1].cast(T.IntegerType()))
    .drop(*metric_cols + ['array_map'])
    .filter('value is not null and disposition is not null')
)

disposition_long = (disposition_long
                    .groupby('call_guid', 'offer')
                    .pivot('disposition')
                    .max('value')            
                    .fillna(0)
)

call = spark.table('prod_oco_team_data_science.gold.call_log_topic')
disposition_long = disposition_long.join(call.select('call_guid', 'cagm', 'start_dtm'), on='call_guid')

eid = (spark.table('prod_oco_team_data_science.gold.eid_to_alternative_id')
       .filter('''id_type = 'cagm' ''')
       .select('enterprise_id', F.col('alternative_id').alias('cagm'))
)
disposition_long = disposition_long.join(eid, on='cagm').drop('cagm')

output_table = disposition_long.select(
    'enterprise_id',
    F.lit('xxx').alias('organization'),
    F.lit('ivr').alias('app'),
    F.lit('source').alias('xxxrx'),
    F.lit('category').alias('call'),
    F.to_date('start_dtm').alias('date'),
    F.col('start_dtm').alias('date_time'),
    'offer',
    'offered', 'accepted', 'success', 'failure'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Specialty Rx web Offer Dispositions
# MAGIC query the adobe clickstream dataset to get the values for dispositions of offers from PSO

# COMMAND ----------

specialty_rx = spark.table('prod_oco_data_analytics.clickstream_xxx.xxxrxbriovarxprod_hit_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carepathways Offer Dispositions
# MAGIC get this from the carepathways schema in our unity catalog

# COMMAND ----------

cpw_offers = spark.table('prod_oco_team_data_science.care_pathways.care_pth_act')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save final output table

# COMMAND ----------

# union multiple tables if necessary

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

# COMMAND ----------


