# Databricks notebook source
import os, json, time, random, sys
from datetime import date
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime
import pytz
from pytz import timezone
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from pyspark.ml.feature import QuantileDiscretizer
import pandas as pd
import numpy as np
from itertools import chain
from functools import partial
import torch
from torch.nn.functional import softmax
from Crypto.Cipher import AES
import hashlib
import binascii
import Padding
from binascii import unhexlify
import base64

ACTION_MAP = {
    0: 'none',
    1: 'autorefill',
    2: 'sms',
    3: 'renewal',
    4: 'schedulerefill',
    5: 'refill',
    6: 'orderstatus',
    7: 'outstandingbalance',
    8: 'address',
    9: 'creditcard',
    10: 'clinicalopportunity',
    11: 'diabeteshold'
}

random.seed(15)
colors = px.colors.qualitative.Alphabet
# random.shuffle(colors)

c = dict(zip(list(ACTION_MAP.values()), colors))

action_mapper = F.create_map([F.lit(x) for x in chain(*ACTION_MAP.items())])

IMPLEMENTED_DATE = "2022-10-01"
FIRST_TRAIN_DATE = "2022-10-28"
CUT_OFF_DATE = date(2022, 6, 1)
PILOT_TFN = '8778896510'

# COMMAND ----------

# mem = spark.table('digital_analytics_bronze.imdm_')
episodes = spark.table('digital_analytics_gold.personalization_training_episodes')
agent_data = spark.table('digital_analytics_gold.personalization_agent_data')
reward_table = spark.table('digital_analytics_gold.personalization_reward_report').filter(f'startdate > "2022-01-01" and metric not in ("pasms")')
pilot_data = reward_table.filter('control_pilot = 0')
control_data = reward_table.filter('control_pilot = 1')
control_size = control_data.count()
pilot_size = pilot_data.count()
control_data = control_data.sample(pilot_size / control_size, seed=42)
model_data = pilot_data.union(control_data)

# COMMAND ----------

# tfn_tab = reward_table.select('gvpsessionid', 'startdate', 'orig_tfn', 'tfn').distinct()
# tfn_tab = tfn_tab.groupby('startdate', 'orig_tfn', 'tfn').agg(F.count('*').alias('volume'))
# tfn_tab = tfn_tab.toPandas()
# # tfn_tab['startdate'] = pd.to_datetime(tfn_tab['startdate']).dt.date
# # tfn_tab = tfn_tab.set_index(pd.DatetimeIndex(tfn_tab['startdate'])).groupby([pd.Grouper(freq='W'), 'orig_tfn', 'tfn'])[['volume']].sum().reset_index()
# tfn_tab = tfn_tab.groupby(['orig_tfn', 'tfn'])['volume'].mean().reset_index()
# display(tfn_tab)

# COMMAND ----------

total_n = reward_table.select('gvpsessionid', 'startdate', 'tfn').distinct().groupby('startdate').agg(F.count('*').alias('n'))
calls_n = reward_table.select('gvpsessionid', 'startdate', 'tfn').distinct().groupby('startdate', 'tfn').agg(F.count('*').alias('n'))

display(total_n.select(F.round(F.avg('n'), 0).alias('average_calls_per_day')))
display(calls_n.groupby('tfn').agg(F.round(F.avg('n'), 0).alias('average_calls_per_day')))

# COMMAND ----------

# MAGIC %md
# MAGIC # Accepted Offers

# COMMAND ----------

rew = reward_table.filter('metric not in ("call", "resolved") and offered = 1').groupby('startdate', 'tfn', 'metric').agg(F.sum('accepted').alias('accepted'), F.sum('offered').alias('offered'))
rew = rew.toPandas()
rew['startdate'] = pd.to_datetime(rew['startdate']).dt.date
rew_cat = rew[(rew['tfn'] == 'pilot')].copy()

display(reward_table.filter('before_after is not null').groupby(['startdate', 'before_after', 'tfn']).agg(F.sum('accepted').alias('accepted')).groupby(['before_after', 'tfn']).agg(F.avg('accepted').alias('avg_accepted')))

fig = px.bar(rew_cat, x="startdate", y="accepted", color="metric", color_discrete_map=c, barmode='stack', title="Accepted Offer Counts")
fig.add_vline(x=IMPLEMENTED_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=IMPLEMENTED_DATE, y=1, yref='paper', text='Implemented')
fig.add_vline(x=FIRST_TRAIN_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=FIRST_TRAIN_DATE, y=1, yref='paper', text='Trained')
fig.show(renderer='databricks')

rewts = rew.set_index(pd.DatetimeIndex(rew['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn'])[['accepted', 'offered']].sum().reset_index()
rewts['startdate'] = pd.to_datetime(rewts['startdate']).dt.date
rewts.sort_values(by='startdate', inplace=True)
rewts['accepted_rate'] = rewts['accepted'] / rewts['offered']

fig = px.line(rewts, x="startdate", y="accepted_rate", color='tfn', title="Accepted Rate")
fig.add_vline(x=IMPLEMENTED_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=IMPLEMENTED_DATE, y=1, yref='paper', text='Implemented')
fig.add_vline(x=FIRST_TRAIN_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=FIRST_TRAIN_DATE, y=1, yref='paper', text='Trained')
fig.show(renderer='databricks')

ob = rew[rew['metric'] == 'outstandingbalance'].copy()
ob = ob.set_index(pd.DatetimeIndex(ob['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn'])[['accepted', 'offered']].sum().reset_index()
ob['startdate'] = pd.to_datetime(ob['startdate']).dt.date
ob.sort_values(by='startdate', inplace=True)
ob['accepted_rate'] = ob['accepted'] / ob['offered']

fig = px.line(ob, x="startdate", y="accepted_rate", color='tfn', title="Outstanding Balance Accepted Rate")
fig.add_vline(x=IMPLEMENTED_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=IMPLEMENTED_DATE, y=1, yref='paper', text='Implemented')
fig.add_vline(x=FIRST_TRAIN_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=FIRST_TRAIN_DATE, y=1, yref='paper', text='Trained')
fig.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC # Repeat Calls within 7 Days

# COMMAND ----------

m = reward_table.filter('metric = "call"').groupby('gvpsessionid', 'cagm', 'tfn', 'startdate', 'xferstatus', 'before_after').agg(F.max('penalty').alias('penalty'))
m = m.withColumnRenamed('penalty', 'repeat_calls')

display(m.filter('xferstatus = "XFER" and before_after is not null').groupby(['startdate', 'before_after', 'tfn']).agg(F.sum('repeat_calls').alias('repeat_calls')).groupby(['before_after', 'tfn']).agg(F.avg('repeat_calls').alias('avg_repeat_calls')))

xfer = reward_table.filter('xferstatus = "XFER" and metric = "call"').groupby('tfn', 'startdate').agg(F.sum('penalty').alias('repeat_calls'), F.count('*').alias('calls'))
xfer = xfer.toPandas()
xfer['startdate'] = pd.to_datetime(xfer['startdate'])
xfer = xfer.set_index(pd.DatetimeIndex(xfer['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn'])[['repeat_calls', 'calls']].sum().reset_index()
xfer['repeat_call_rate'] = xfer['repeat_calls'] / xfer['calls']
xfer['startdate'] = pd.to_datetime(xfer['startdate']).dt.date

m = m.groupby('tfn', 'startdate').agg(F.sum('repeat_calls').alias('repeat_calls'), F.count('*').alias('calls'))
m = m.toPandas()
m['startdate'] = pd.to_datetime(m['startdate'])

m = m.set_index(pd.DatetimeIndex(m['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn'])[['repeat_calls', 'calls']].sum().reset_index()
m['repeat_call_rate'] = m['repeat_calls'] / m['calls']
m['startdate'] = pd.to_datetime(m['startdate']).dt.date

m.sort_values(by=['startdate', 'tfn'], inplace=True)

fig = px.line(m, x="startdate", y="repeat_call_rate", color='tfn', title="Repeat Call Rate")
fig.add_vline(x=IMPLEMENTED_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=IMPLEMENTED_DATE, y=1, yref='paper', text='Implemented')
fig.add_vline(x=FIRST_TRAIN_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=FIRST_TRAIN_DATE, y=1, yref='paper', text='Trained')
fig.show(renderer='databricks')

fig = px.line(xfer, x="startdate", y="repeat_call_rate", color='tfn', title="Repeat Call Rate (Xfer)")
fig.add_vline(x=IMPLEMENTED_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=IMPLEMENTED_DATE, y=1, yref='paper', text='Implemented')
fig.add_vline(x=FIRST_TRAIN_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=FIRST_TRAIN_DATE, y=1, yref='paper', text='Trained')
fig.show(renderer='databricks')

# COMMAND ----------

m = reward_table.filter('metric = "resolved"').groupby('gvpsessionid', 'cagm', 'tfn', 'startdate', 'xferstatus', 'before_after').agg(F.max('penalty').alias('penalty'))
m = m.withColumnRenamed('penalty', 'resolved_calls')

display(m.filter('before_after is not null').groupby(['startdate', 'before_after', 'tfn']).agg(F.sum('resolved_calls').alias('resolved_calls')).groupby(['before_after', 'tfn']).agg(F.avg('resolved_calls').alias('avg_resolved_calls')))

m = m.groupby('tfn', 'startdate').agg(F.sum('resolved_calls').alias('resolved_calls'), F.count('*').alias('calls'))
m = m.toPandas()
m['startdate'] = pd.to_datetime(m['startdate'])

m = m.set_index(pd.DatetimeIndex(m['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn'])[['resolved_calls', 'calls']].sum().reset_index()
m['resolved_rate'] = m['resolved_calls'] / m['calls']
m['startdate'] = pd.to_datetime(m['startdate']).dt.date

m.sort_values(by=['startdate', 'tfn'], inplace=True)

fig = px.line(m, x="startdate", y="resolved_rate", color='tfn', title="Resolved Call Rate")
fig.add_vline(x=IMPLEMENTED_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=IMPLEMENTED_DATE, y=1, yref='paper', text='Implemented')
fig.add_vline(x=FIRST_TRAIN_DATE, line_width=3, line_dash="dash", line_color="green")
fig.add_annotation(x=FIRST_TRAIN_DATE, y=1, yref='paper', text='Trained')
fig.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC # Difference in Difference Comparison

# COMMAND ----------

call_diff = reward_table.filter('metric = "call" and startdate >= "2022-06-01" and xferstatus = "XFER"').groupby('gvpsessionid', 'cagm', 'tfn', 'startdate', 'before_after').agg(F.max('penalty').alias('penalty'))
call_diff = call_diff.withColumnRenamed('penalty', 'repeat_calls')

call_diff = call_diff.groupby('tfn', 'startdate', 'before_after').agg(F.sum('repeat_calls').alias('repeat_calls'), F.count('*').alias('calls'))
call_diff = call_diff.toPandas()
call_diff = call_diff.set_index(pd.DatetimeIndex(call_diff['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn', 'before_after'])[['repeat_calls', 'calls']].sum().reset_index()
call_diff['startdate'] = pd.to_datetime(call_diff['startdate']).dt.date

after_rew_diff = call_diff[call_diff['before_after'] == 0].copy()
after_rew_diff = after_rew_diff.groupby('tfn')[['repeat_calls', 'calls']].mean().reset_index()
after_rew_diff['repeat_calls'] = after_rew_diff['repeat_calls'].round(0)
after_rew_diff['calls'] = after_rew_diff['calls'].round(0)
after_rew_diff['call_rate'] = after_rew_diff['repeat_calls'] / after_rew_diff['calls']

before_rew_diff = call_diff[call_diff['before_after'] == 1].copy()
before_rew_diff = before_rew_diff.groupby('tfn')[['repeat_calls', 'calls']].mean().reset_index()
before_rew_diff['before_repeat_calls'] = before_rew_diff['repeat_calls'].round(0)
before_rew_diff['before_calls'] = before_rew_diff['calls'].round(0)
before_rew_diff['before_call_rate'] = before_rew_diff['before_repeat_calls'] / before_rew_diff['before_calls']
before_rew_diff.drop(['repeat_calls', 'calls'], axis=1, inplace=True)

before_after_diff = after_rew_diff.merge(before_rew_diff, on='tfn')
before_after_diff['diff'] = before_after_diff.at[1, 'before_call_rate'] - before_after_diff.at[0, 'before_call_rate']
before_after_diff['expected'] = before_after_diff.at[0, 'call_rate'] + before_after_diff['diff']
before_after_diff['effect'] = before_after_diff['call_rate'] + before_after_diff['expected']
before_after_diff['effect_pct'] = before_after_diff['call_rate'] / before_after_diff['expected']

plot_data = [{'group': 'control', 'pre-smartagent': before_after_diff.at[0, 'before_call_rate'], 'post-smartagent': before_after_diff.at[0, 'call_rate'],},
            {'group': 'pilot', 'pre-smartagent': before_after_diff.at[1, 'before_call_rate'], 'post-smartagent': before_after_diff.at[1, 'call_rate'],},
            {'group': 'expected', 'pre-smartagent': before_after_diff.at[1, 'before_call_rate'], 'post-smartagent': before_after_diff.at[1, 'expected'],}]
plot_data = pd.DataFrame.from_records(plot_data)

did = ((plot_data.at[0, 'pre-smartagent'] - plot_data.at[0, 'post-smartagent']) - 
        (plot_data.at[1, 'pre-smartagent'] - plot_data.at[1, 'post-smartagent']))
print(f'Difference in Difference: {did:,}')
print(f'Percent repeat calls below expected: {before_after_diff.at[1, "effect_pct"] * 100:.2f}%')

before_after_diff['total_effect'] = ((before_after_diff['call_rate'] + did) * before_after_diff['calls']) - before_after_diff['repeat_calls']
display(before_after_diff[['tfn', 'before_repeat_calls', 'before_calls', 'before_call_rate', 'repeat_calls', 'calls', 'call_rate', 'total_effect']])

plot_data = pd.melt(plot_data, id_vars=['group'], value_vars=['pre-smartagent', 'post-smartagent'], var_name='before_after', value_name='repeat calls')
fig = px.line(plot_data, x="before_after", y="repeat calls", color='group', title="Implementation of agent model reduced repeat calls by ~2%")
fig.show(renderer='databricks')
display(plot_data)

pen_win = Window().partitionBy('gvpsessionid').orderBy('metric').rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
action_diff = reward_table.filter('startdate >= "2022-06-01" and xferstatus = "XFER" and before_after is not null')
action_diff = action_diff.withColumn('penalty', F.max('penalty').over(pen_win) * F.col('accepted'))
action_diff = action_diff.filter('metric not in ("call", "resolved")').groupby('tfn', 'startdate', 'before_after', 'metric').agg(F.sum('penalty').alias('repeat_calls'), F.sum('accepted').alias('accepted'))
action_diff = action_diff.toPandas()
action_diff = action_diff.set_index(pd.DatetimeIndex(action_diff['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn', 'before_after', 'metric'])[['repeat_calls', 'accepted']].sum().reset_index()
action_diff['before_after'] = action_diff['before_after'].map({1: 'before', 0: 'after'})
action_diff = pd.pivot_table(action_diff, values=['repeat_calls', 'accepted'], columns=['before_after'], index=['tfn', 'metric'], aggfunc=np.sum)
action_diff.columns = ["_".join(tup) for tup in action_diff.columns.to_flat_index()]
action_diff.reset_index(inplace=True)
action_diff['before_call_rate'] = action_diff['repeat_calls_before'] / action_diff['accepted_before']
action_diff['after_call_rate'] = action_diff['repeat_calls_after'] / action_diff['accepted_after']
# action_diff = action_diff[['tfn', 'metric', 'before_call_rate', 'after_call_rate']]
display(action_diff.sort_values(by=['tfn', 'accepted_after']), ascending=['True', 'False'])

# COMMAND ----------

acc_diff = reward_table.filter('metric not in ("call", "resolved") and offered = 1 and before_after is not null')
acc_diff = acc_diff.groupby('tfn', 'startdate', 'before_after').agg(F.sum('accepted').alias('accepted'), F.sum('offered').alias('offered'))
acc_diff = acc_diff.toPandas()
acc_diff = acc_diff.set_index(pd.DatetimeIndex(acc_diff['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn', 'before_after'])[['accepted', 'offered']].sum().reset_index()
acc_diff['startdate'] = pd.to_datetime(acc_diff['startdate']).dt.date

after_rew_diff = acc_diff[acc_diff['before_after'] == 0].copy()
after_rew_diff = after_rew_diff.groupby('tfn')[['accepted', 'offered']].mean().reset_index()
after_rew_diff['accepted'] = after_rew_diff['accepted'].round(0)
after_rew_diff['offered'] = after_rew_diff['offered'].round(0)
after_rew_diff['accepted_rate'] = after_rew_diff['accepted'] / after_rew_diff['offered']

before_rew_diff = acc_diff[acc_diff['before_after'] == 1].copy()
before_rew_diff = before_rew_diff.groupby('tfn')[['accepted', 'offered']].mean().reset_index()
before_rew_diff['before_accepted'] = before_rew_diff['accepted'].round(0)
before_rew_diff['before_offered'] = before_rew_diff['offered'].round(0)
before_rew_diff['before_accepted_rate'] = before_rew_diff['before_accepted'] / before_rew_diff['before_offered']
before_rew_diff.drop(['accepted', 'offered'], axis=1, inplace=True)

before_after_diff = after_rew_diff.merge(before_rew_diff, on='tfn')
before_after_diff['diff'] = before_after_diff.at[1, 'before_accepted_rate'] - before_after_diff.at[0, 'before_accepted_rate']
before_after_diff['counterfactual'] = before_after_diff.at[0, 'accepted_rate'] + before_after_diff['diff']
before_after_diff['effect'] = before_after_diff['accepted_rate'] - before_after_diff['counterfactual']
before_after_diff['effect_pct'] = before_after_diff['accepted_rate'] / before_after_diff['counterfactual']

plot_data = [{'group': 'control',
              'pre-smartagent': before_after_diff.at[0, 'before_accepted_rate'],
              'post-smartagent': before_after_diff.at[0, 'accepted_rate'],
             },
            {'group': 'pilot',
             'pre-smartagent': before_after_diff.at[1, 'before_accepted_rate'],
             'post-smartagent': before_after_diff.at[1, 'accepted_rate'],
            },
            {'group': 'counterfactual',
             'pre-smartagent': before_after_diff.at[1, 'before_accepted_rate'],
             'post-smartagent': before_after_diff.at[1, 'counterfactual'],
            }]

plot_data = pd.DataFrame.from_records(plot_data)
did = ((plot_data.at[0, 'pre-smartagent'] - plot_data.at[0, 'post-smartagent']) - 
        (plot_data.at[1, 'pre-smartagent'] - plot_data.at[1, 'post-smartagent']))
print(f'Difference in Difference: {did:,}')
print(f'Percent accepted offers above expected: {before_after_diff.at[1, "effect_pct"] * 100:.2f}%')

before_after_diff['total_effect'] = ((before_after_diff['accepted_rate'] + did) * before_after_diff['offered']) - before_after_diff['accepted']
display(before_after_diff[['tfn', 'before_accepted', 'before_offered', 'before_accepted_rate', 'accepted', 'offered', 'accepted_rate', 'total_effect']])

plot_data = pd.melt(plot_data, id_vars=['group'], value_vars=['pre-smartagent', 'post-smartagent'], var_name='before_after', value_name='accepted offers')
fig = px.line(plot_data, x="before_after", y="accepted offers", color='group', title="Accepted Offers Counterfactual Analysis")
fig.show(renderer='databricks')
display(plot_data)

action_diff = reward_table.filter('metric not in ("call", "resolved") and offered = 1 and before_after is not null')
action_diff = action_diff.groupby('tfn', 'startdate', 'before_after', 'metric').agg(F.sum('offered').alias('offered'), F.sum('accepted').alias('accepted'))
action_diff = action_diff.toPandas()
action_diff = action_diff.set_index(pd.DatetimeIndex(action_diff['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn', 'before_after', 'metric'])[['offered', 'accepted']].sum().reset_index()
action_diff['before_after'] = action_diff['before_after'].map({1: 'before', 0: 'after'})
action_diff = pd.pivot_table(action_diff, values=['offered', 'accepted'], columns=['before_after'], index=['tfn', 'metric'], aggfunc=np.sum)
action_diff.columns = ["_".join(tup) for tup in action_diff.columns.to_flat_index()]
action_diff.reset_index(inplace=True)
action_diff['accepted_rate_before'] = action_diff['accepted_before'] / action_diff['offered_before']
action_diff['accepted_rate_after'] = action_diff['accepted_after'] / action_diff['offered_after']
# action_diff = action_diff[['tfn', 'metric', 'before_call_rate', 'after_call_rate']]
display(action_diff.sort_values(by=['tfn', 'accepted_after']), ascending=['True', 'False'])

# COMMAND ----------

call_diff = reward_table.filter('metric = "resolved" and startdate >= "2022-06-01"')\
                .groupby('gvpsessionid', 'cagm', 'tfn', 'startdate', 'before_after').agg(F.max('penalty').alias('penalty'))
call_diff = call_diff.withColumnRenamed('penalty', 'resolved_calls')

call_diff = call_diff.groupby('tfn', 'startdate', 'before_after').agg(F.sum('resolved_calls').alias('resolved_calls'), F.count('*').alias('calls'))
call_diff = call_diff.toPandas()
call_diff = call_diff.set_index(pd.DatetimeIndex(call_diff['startdate'])).groupby([pd.Grouper(freq='W'), 'tfn', 'before_after'])[['resolved_calls', 'calls']].sum().reset_index()
call_diff['startdate'] = pd.to_datetime(call_diff['startdate']).dt.date

after_rew_diff = call_diff[call_diff['before_after'] == 0].copy()
after_rew_diff = after_rew_diff.groupby('tfn')[['resolved_calls', 'calls']].mean().reset_index()
after_rew_diff['resolved_calls'] = after_rew_diff['resolved_calls'].round(0)
after_rew_diff['calls'] = after_rew_diff['calls'].round(0)
after_rew_diff['resolved_rate'] = after_rew_diff['resolved_calls'] / after_rew_diff['calls']

before_rew_diff = call_diff[call_diff['before_after'] == 1].copy()
before_rew_diff = before_rew_diff.groupby('tfn')[['resolved_calls', 'calls']].mean().reset_index()
before_rew_diff['before_resolved_calls'] = before_rew_diff['resolved_calls'].round(0)
before_rew_diff['before_calls'] = before_rew_diff['calls'].round(0)
before_rew_diff['before_resolved_rate'] = before_rew_diff['before_resolved_calls'] / before_rew_diff['before_calls']
before_rew_diff.drop(['resolved_calls', 'calls'], axis=1, inplace=True)

before_after_diff = after_rew_diff.merge(before_rew_diff, on='tfn')
before_after_diff['diff'] = before_after_diff.at[1, 'before_resolved_rate'] - before_after_diff.at[0, 'before_resolved_rate']
before_after_diff['counterfactual'] = before_after_diff.at[0, 'resolved_rate'] + before_after_diff['diff']
before_after_diff['effect'] = before_after_diff['resolved_rate'] + before_after_diff['counterfactual']
before_after_diff['effect_pct'] = before_after_diff['resolved_rate'] / before_after_diff['counterfactual']

plot_data = [{'group': 'control', 'pre-smartagent': before_after_diff.at[0, 'before_resolved_rate'], 'post-smartagent': before_after_diff.at[0, 'resolved_rate'],},
            {'group': 'pilot', 'pre-smartagent': before_after_diff.at[1, 'before_resolved_rate'], 'post-smartagent': before_after_diff.at[1, 'resolved_rate'],},
            {'group': 'counterfactual', 'pre-smartagent': before_after_diff.at[1, 'before_resolved_rate'], 'post-smartagent': before_after_diff.at[1, 'counterfactual'],}]
plot_data = pd.DataFrame.from_records(plot_data)

did = ((plot_data.at[0, 'pre-smartagent'] - plot_data.at[0, 'post-smartagent']) - 
        (plot_data.at[1, 'pre-smartagent'] - plot_data.at[1, 'post-smartagent']))
print(f'Difference in Difference: {did:,}')
print(f'Percent repeat calls below expected: {before_after_diff.at[1, "effect_pct"] * 100:.2f}%')

before_after_diff['total_effect'] = ((before_after_diff['resolved_rate'] + did) * before_after_diff['calls']) - before_after_diff['resolved_calls']
display(before_after_diff[['tfn', 'before_resolved_calls', 'before_calls', 'before_resolved_rate', 'resolved_calls', 'calls', 'resolved_rate', 'total_effect']])

plot_data = pd.melt(plot_data, id_vars=['group'], value_vars=['pre-smartagent', 'post-smartagent'], var_name='before_after', value_name='resolved calls')
fig = px.line(plot_data, x="before_after", y="resolved calls", color='group', title="Resolved Calls Counterfactual Analysis")
fig.show(renderer='databricks')
display(plot_data)

pen_win = Window().partitionBy('gvpsessionid').orderBy('metric').rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
action_diff = reward_table.filter('startdate >= "2022-06-01" and xferstatus = "XFER" and before_after is not null')
action_diff = action_diff.withColumn('penalty', F.max('penalty').over(pen_win) * F.col('accepted'))
action_diff = action_diff.filter('metric not in ("call", "resolved")').groupby('tfn', 'startdate', 'before_after', 'metric')\
                .agg(F.sum('penalty').alias('resolved_calls'), F.sum('accepted').alias('accepted'))
action_diff = action_diff.toPandas()
action_diff = action_diff.set_index(pd.DatetimeIndex(action_diff['startdate']))\
                .groupby([pd.Grouper(freq='W'), 'tfn', 'before_after', 'metric'])[['resolved_calls', 'accepted']].sum().reset_index()
action_diff['before_after'] = action_diff['before_after'].map({1: 'before', 0: 'after'})
action_diff = pd.pivot_table(action_diff, values=['resolved_calls', 'accepted'], columns=['before_after'], index=['tfn', 'metric'], aggfunc=np.sum)
action_diff.columns = ["_".join(tup) for tup in action_diff.columns.to_flat_index()]
action_diff.reset_index(inplace=True)
action_diff['before_resolved_rate'] = action_diff['resolved_calls_before'] / action_diff['accepted_before']
action_diff['after_resolved_rate'] = action_diff['resolved_calls_after'] / action_diff['accepted_after']
# action_diff = action_diff[['tfn', 'metric', 'before_call_rate', 'after_call_rate']]
display(action_diff.sort_values(by=['tfn', 'accepted_after']), ascending=['True', 'False'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Agent Interactions

# COMMAND ----------

value_dist = agent_data.filter('q_values is not null and best_action_dsc not in ("clinicalopportunity", "creditcard", "address")').select('best_action', 'best_action_dsc', 'q_values')
value_dist = value_dist.withColumn('q_value', F.col('q_values')[F.col('best_action')])
value_dist = value_dist.withColumnRenamed('best_action_dsc', 'action_nm').toPandas()
value_dist['q_value'] = value_dist['q_value'].clip(upper=value_dist['q_value'].std() * 2, lower=value_dist['q_value'].std() * -2)

value_rank = value_dist.groupby('action_nm')['q_value'].median().reset_index()
value_rank['rank'] = value_rank['q_value'].rank(method='dense', ascending=False)
value_rank.sort_values(by='rank', inplace=True)
display(value_rank)

fig = px.histogram(value_dist, x="q_value", color='action_nm', marginal='rug',
                   color_discrete_map=c, title="Q Value Distribution")
fig.show(renderer='databricks')

## for a more detailed look at each distribution
fig = px.histogram(value_dist, x="q_value", facet_row='action_nm', color='action_nm',
                   color_discrete_map=c, facet_col_wrap=1, title="Q Value Detail Distributions")
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.for_each_annotation(lambda a: a.update(textangle=0))

fig.update_layout(
#     autosize=False,
    width=900,
    height=2000,
    legend_x=2)

fig.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC # Fairness

# COMMAND ----------

pwd = dbutils.secrets.get(scope="imdm_encryption_keys_kv", key="imdm-key")
iv = dbutils.secrets.get(scope="imdm_encryption_keys_kv", key="imdm-iv")

def imdm_decrypt(ciphertext, password, ival):
        key = password.encode()
        iv = ival.encode()
        encobj = AES.new(key, AES.MODE_CBC, iv)
        plaintext = encobj.decrypt(base64.b64decode(ciphertext.encode()))
        return(Padding.removePadding(plaintext.decode(), mode=0))

decrypt_pyfunc = udf(imdm_decrypt, StringType())

# COMMAND ----------

mem_indiv = spark.table('digital_analytics_bronze.imdm_d_master_individual')

testdf = testdf.withColumn('date_of_birth', decrypt_pyfunc('INDV_DT_OF_BRTH', lit(pwd), lit(iv)))
testdf.display()
