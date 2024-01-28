
import os, json, shutil
import numpy as np
import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.keyvault.secrets import SecretClient

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from agent.config import (ENV_CONFIG, KEYVAULT_NAME,
                          AML_CONFIG, SERVICE_PRINCIPAL,
                          DB_TOKEN_SECRET, ACTION_MAP)


for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])
secret_client = SecretClient(vault_url=f'https://{KEYVAULT_NAME}.vault.azure.net/', credential=credential)

p = os.path.abspath(os.getcwd())
BATCH_PATH = os.path.join(p, 'sample_batches')

MLFLOW_URL = os.environ['MLFLOW_TRACKING_URI']
DB_TOKEN = secret_client.get_secret(DB_TOKEN_SECRET).value


def get_dataset(train_data_path):
    training_data = pd.read_parquet(train_data_path)

    keeps = {'agent_index', 'eps_id', 't', 'actions', 'obs', 'rewards',
             'dones', 'new_obs', 'action_prob', 'action_logp', 'mask', 'new_mask'}
    drops = set(training_data.columns) - keeps
    training_data.drop(drops, axis=1, inplace=True)
    return training_data


def write_batch(write_path, data):
    writer = JsonWriter(write_path)
    batch_builder = SampleBatchBuilder()
    prep = get_preprocessor(ENV_CONFIG['observation_space'])(ENV_CONFIG['observation_space'])
    for id, eps_df in data.groupby('eps_id'):
        eps = eps_df.to_dict('records')
        for t in eps:
            t['obs'] = prep.transform({'real_obs': np.array(t['obs']).tolist(),
                                       'mask': np.array(t['mask']).tolist()})
            t['new_obs'] = prep.transform({'real_obs': np.array(t['new_obs']).tolist(),
                                           'mask': np.array(t['new_mask']).tolist()})
            t.pop('mask')
            t.pop('new_mask')
            batch_builder.add_values(**t)
    batch = batch_builder.build_and_reset()
    writer.write(batch)


def gather_sample_batches(train_data_path, train_size=0.7):
    if os.path.exists(BATCH_PATH):
        shutil.rmtree(BATCH_PATH)
    os.mkdir(os.path.join(BATCH_PATH))
    os.mkdir(os.path.join(BATCH_PATH, 'train'))
    os.mkdir(os.path.join(BATCH_PATH, 'eval'))

    training_data = get_dataset(train_data_path)

    eps_id_list = training_data['eps_id'].unique().tolist()
    np.random.seed(42)
    train_eps = np.random.choice(eps_id_list,
                                 int(len(eps_id_list) * train_size), replace=False)
    x = training_data[training_data['eps_id'].isin(train_eps)].copy()
    y = training_data[~training_data['eps_id'].isin(train_eps)].copy()

    write_batch(os.path.join(BATCH_PATH, 'train'), x)
    write_batch(os.path.join(BATCH_PATH, 'eval'), y)


def validate_model(train_data_path, agent, ckpt):
    training_data = get_dataset(train_data_path)
    agent.restore(ckpt)

    records = []
    for r in training_data.itertuples():
        x = {'mask': r.mask, 'real_obs': r.obs}
        action_dict = agent.compute_single_action(x, full_fetch=True, explore=False)
        action = int(action_dict[0])
        q_values = action_dict[2]['q_values']
        orig_action = r.actions
        app_action_map = {**ACTION_MAP['common'], **ACTION_MAP[r.app]}
        records.append({'action': action, 'action_dsc': app_action_map[action], 'obs': r.obs, 'mask': r.mask,
                        'orig_action': orig_action, 'orig_action_dsc': app_action_map[orig_action],
                        'value': q_values[action]})

    df = pd.DataFrame.from_records(records)
    orig_df = df.copy().drop(['action', 'action_dsc'], axis=1)\
        .rename(columns={'orig_action': 'action', 'orig_action_dsc': 'action_dsc'})

    def f(x):
        d = []
        d.append(len(x))
        d.append(sum(x['value']))
        d.append(x['value'].mean())
        return pd.Series(d, index=['new_dist', 'total_value', 'mean_value'])

    agg = df.groupby(['action', 'action_dsc']).apply(f).reset_index()
    orig_agg = orig_df.groupby(['action', 'action_dsc']).size().reset_index(name='orig_dist')
    agg = agg.merge(orig_agg, how='left', on=['action', 'action_dsc']).sort_values(by='action')

    assert all([d != 0 for d in agg['new_dist'].tolist()])
    return df, agg


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def clean_json(c):
    ''' ask forgivness, not permission :D '''
    keys_to_delete = []
    for key, value in c.items():
        if not is_jsonable(value):
            #byebye
            keys_to_delete.append(key)

    c = {k: v for k, v in c.items() if k not in keys_to_delete}
    return c

