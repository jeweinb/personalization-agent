
import os
import gym
import numpy as np
from copy import deepcopy
from yaml import full_load

dir_path = os.path.dirname(os.path.realpath(__file__))


AML_CONFIG = {
    "subscription_id": "",
    "resource_group": "",
    "workspace_name": "",
}

SERVICE_PRINCIPAL = {
    'AZURE_CLIENT_ID': '',
    'AZURE_TENANT_ID': '',
    'AZURE_CLIENT_SECRET': '',
}
for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

VNET_NAME = 'xxx'
SUBNET_NAME = 'xxx'
TRAIN_SUBNET_NAME = 'xxx'
KEYVAULT_NAME = 'xxx'
CONTAINER_REGISTRY = 'xxxr.azurecr.io'
AML_TRAIN_ENV = 'custom-train-env-v2'
AML_DEPLOY_ENV = 'curated-env-v2'
GOLD_STORAGE = 'xxx'

## Databricks connection
DB_HOST = "xxx"
DB_PATH = "xxx"
DB_TOKEN_SECRET = "xxx"
DB_WORKSPACE = 'xxx'
DB_CLUSTER_ID = 'xxx'

## Standalone app deployment
MODEL_ADDRESS = 'http://localhost:8000/ivr-agent'
HOST = '0.0.0.0'
PORT = 8081

## AML compute and docker env
COMPUTES = [
    {'name': 'lrg-cpu-cluster', 'size': 'STANDARD_F64S_V2', 'max': 1},
    {'name': 'gpu-cluster', 'size': 'Standard_NC6s_v3', 'max': 4}
]

IMG_ENVS = [
    {'name': 'sdkv2-agent-env-v2',
     'base': 'xxx.azurecr.io/sdkv2_agent_img:latest'},
    {'name': 'agent-inference-env',
     'base': 'xxx.azurecr.io/sdkv2_agent_img_inference:latest'},
]

## Agent environment app_configs
AVAILABLE_MODELS = {
    'conf1': 'conf1.yaml',
    'conf2': 'conf2.yaml'
}

DEPLOY_MODEL = 'conf1'

with open(os.path.join(dir_path, 'app_configs', AVAILABLE_MODELS[DEPLOY_MODEL]), 'r') as f:
    configs = full_load(f)

DATA_STORE = configs['app_data']['datastore']
GRAD_ACCUM_BATCHES = configs['grad_accum_batches']
ENCODER_BATCH_SIZE = configs['encoder_batch_size']
ENCODER_RESUME_TRAINING = configs['encoder_resume']
TRAIN_COMPUTE = configs['train_compute']
NUM_NODES = configs['num_nodes']
NUM_DEVICES_PER_NODE = configs['num_devices_per_node']
ACTION_MAP = configs['action_space']
ACTION_DIM = configs['action_dim']
STATE_DIM = configs['state_dim']
STATE_MAX_LEN = configs['state_max_len']
APP_DATA_MAP = configs['app_data']
TOPN_ACTIONS = configs['topn_actions']
CHANCE_OF_NO_ACTION = configs['chance_no_action']

COLD_START = 'xxx'  # use None to signal that we don't need cold start

ENV_CONFIG = {
    'env': None,
    'observation_space': gym.spaces.Dict({
                    'mask': gym.spaces.Box(0, 1, shape=(ACTION_DIM,)),
                    'real_obs': gym.spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,))}),
    'action_space': gym.spaces.Discrete(ACTION_DIM),
}

AGENT_CONFIG = {
    'model': {'custom_model': 'maskable_dqn',
              'custom_model_config': {'action_dim': ACTION_DIM, 'true_obs_shape': (STATE_DIM,)}},
    "framework": "torch",
    "num_workers": 0,
    "num_gpus": 0,
    **ENV_CONFIG,
    "hiddens": [],
    "dueling": False,
    'explore': True,
    "exploration_config": {
        'type': 'EpsilonGreedy',
        'initial_epsilon': .1,
        'final_epsilon': .1,
        'epsilon_timesteps': 1000,
    }
}

if not isinstance(COLD_START, str):
    COLD_START_AGENT_CONFIG = None
elif isinstance(COLD_START, str):
    COLD_START_AGENT_CONFIG = deepcopy(AGENT_CONFIG)
    cold_start_action = sum([len(v) for v in ACTION_MAP.values()])
    COLD_START_AGENT_CONFIG['model']['custom_model_config']['action_dim'] = cold_start_action
    COLD_START_AGENT_CONFIG['observation_space']['mask'] = gym.spaces.Box(0, 1, shape=(cold_start_action,))
    COLD_START_AGENT_CONFIG['action_space'] = gym.spaces.Discrete(cold_start_action)
