# Databricks notebook source
import os, shutil
import pytz
from dateutil.parser import parse
from datetime import datetime, timedelta
import logging
import pyspark.sql.functions as F
import pyspark.sql.types as T

def is_date(string, fuzzy=False):
    try: 
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False
    

PATH = '/dbfs/mnt/ocep/eureka-gold/personalization'
ORIG_FOLDER = 'agent_raw_data'
PROCESSED_FOLDER = 'processed_agent_data'


tz = pytz.timezone('US/Central')
time = datetime.now(tz)
dt_string = time.strftime("%Y-%m-%d %H:%M:%S")

if not os.path.exists(os.path.join(PATH, 'processed_agent_data_logs')):
    os.mkdir(os.path.join(PATH, 'processed_agent_data_logs'))
    
logging.basicConfig(filename=os.path.join(PATH, 'processed_agent_data_logs', f'{dt_string}_process_agent_data_raw.log'),
                    format='%(asctime)s | %(levelname)s: %(message)s',
                    level=logging.NOTSET)

# COMMAND ----------

files = list(os.listdir(os.path.join(PATH, ORIG_FOLDER)))

for i, f in enumerate(files):
    date_part = f.split('_')[1].split('.')[0].split(' ')[0]
    print(date_part)
    
    f_orig = f
    f = f.replace(':', '') # because spark dislikes colon in file names
    os.rename(os.path.join(PATH, ORIG_FOLDER, f_orig),
              os.path.join(PATH, ORIG_FOLDER, f))

    is_directory = os.path.isdir(os.path.join(PATH, ORIG_FOLDER, f))
    
    if is_date(date_part) and not is_directory:
        if not os.path.exists(os.path.join(PATH, PROCESSED_FOLDER)):
            os.mkdir(os.path.join(PATH, PROCESSED_FOLDER))
            logging.info(f'creating directory: {os.path.join(PATH, PROCESSED_FOLDER)}')

        if not os.path.exists(os.path.join(PATH, PROCESSED_FOLDER, date_part)):
            os.mkdir(os.path.join(PATH, PROCESSED_FOLDER, date_part))
            logging.info(f'creating partition {date_part}: {os.path.join(PATH, PROCESSED_FOLDER, date_part)}')

        if not os.path.exists(os.path.join(PATH, PROCESSED_FOLDER, date_part, f)):
            shutil.copy(os.path.join(PATH, ORIG_FOLDER, f), os.path.join(PATH, PROCESSED_FOLDER, date_part))
            logging.info(f'file {i:,} copy success: {os.path.join(PATH, PROCESSED_FOLDER, date_part, f)}')
    else:
        logging.warning(f'{f} is not in the proper format... skipping file')
        
    if is_directory:
        shutil.rmtree(os.path.join(PATH, ORIG_FOLDER, f))
        logging.warning(f'directory {i:,} remove success: {os.path.join(PATH, ORIG_FOLDER, f)}')
    else:
        os.remove(os.path.join(PATH, ORIG_FOLDER, f))
        logging.info(f'file {i:,} remove success: {os.path.join(PATH, ORIG_FOLDER, f)}')
        
    print(f'item {i} processed: {f}')

# COMMAND ----------

schema = T.StructType([
  T.StructField('app', T.StringType(),True),
  T.StructField('implementation', T.StringType(),True),
  T.StructField('id', T.StringType(),True),
  T.StructField('id_type', T.StringType(),True),
  T.StructField('enterprise_id',T.StringType(),True),
  T.StructField('time',T.TimestampType(),True),
  T.StructField('t',T.StringType(),True),
  T.StructField('eps_id',T.StringType(),True),
  T.StructField('agent_index',T.IntegerType(),True),
  T.StructField('obs',T.ArrayType(T.DoubleType()),True),
  T.StructField('actions',T.ArrayType(T.IntegerType()),True),
  T.StructField('action_prob',T.DoubleType(),True),
  T.StructField('action_logp',T.DoubleType(),True),
  T.StructField('logits',T.ArrayType(T.DoubleType()),True),
  T.StructField('q_values',T.ArrayType(T.DoubleType()),True),
  T.StructField('probs',T.ArrayType(T.DoubleType()),True),
  T.StructField('mask',T.ArrayType(T.IntegerType()),True),
  T.StructField('rewards',T.DoubleType(),True),
  T.StructField('prev_actions',T.ArrayType(T.IntegerType()),True),
  T.StructField('prev_rewards',T.DoubleType(),True),
  T.StructField('dones',T.IntegerType(),True),
  T.StructField('infos',T.StringType(),True),
  T.StructField('new_obs',T.ArrayType(T.DoubleType()),True),
  T.StructField('raw_state',T.ArrayType(T.StringType()),True),
  T.StructField('raw_state_str',T.ArrayType(T.StringType()),True),
  T.StructField('raw_action_str',T.ArrayType(T.StringType()),True),
])

files = list(os.listdir(os.path.join(PATH, PROCESSED_FOLDER)))
files = [datetime.strptime(f, "%Y-%m-%d").date() for f in files]

try:
    existing = spark.table('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw')\
            .select('partition_field').distinct()\
            .orderBy('partition_field').toPandas()['partition_field'].tolist()
except:
    existing = []

diff = set(files) - set(existing)

for f in sorted(diff):
    print(f'processing partition: {f}')
    try:
        agent_data = spark.read.json(f'/mnt/ocep/eureka-gold/personalization/processed_agent_data/{f.strftime("%Y-%m-%d")}/*.json', schema=schema)
        agent_data = agent_data.withColumn('partition_field', F.to_date('time'))
        (agent_data.write.format("delta")
        .partitionBy('partition_field')
        .mode('append')
        .option('overwriteSchema', 'true')
        .option("mergeSchema", "true")
        .saveAsTable('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw')
        )
    except Exception as e:
        print(e)
        continue

# COMMAND ----------

# REPROCESS PREVIOUS DAYS TO CAPTURE EXTRA RECORDS
end_date = datetime.today().date()
end_date = end_date.strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=2)).date()
start_date_str = start_date.strftime("%Y-%m-%d")

files = [f for f in files if f >= start_date]
for fi in files:
    partition_str = fi.strftime("%Y-%m-%d")
    
    agent_data = spark.read.json(f'/mnt/ocep/eureka-gold/personalization/processed_agent_data/{partition_str}/*.json', schema=schema)
    # agent_data = agent_data.withColumn('actions', F.array(F.col('actions').cast('int')))
    agent_data = agent_data.withColumn('partition_field', F.to_date('time'))

    filter_clause = f"partition_field = cast('{partition_str}' as date)"
    
    (agent_data.write.format("delta")
     .partitionBy('partition_field')
     .mode('overwrite')
     .option('overwriteSchema', 'true')
     .option("mergeSchema", "true")
     .option("replaceWhere", filter_clause)
     .saveAsTable('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw')
    )

# COMMAND ----------

test = spark.table('prod_oco_team_data_science.personalization_agent.personalization_agent_data_raw')
display(test.select(F.max('partition_field').alias('end'),
                    F.min('partition_field').alias('start'),
                    F.count('*').alias('n')))

# COMMAND ----------

logging.shutdown()
