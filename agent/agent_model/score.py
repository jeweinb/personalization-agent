import json, shutil, os, sys
sys.path.append('/var/azureml-app')

import io
import pytz
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from ray.rllib.algorithms.dqn import DQN
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from agent.config import (AGENT_CONFIG, COLD_START_AGENT_CONFIG, APP_DATA_MAP,
                          ACTION_MAP, COLD_START, TOPN_ACTIONS, ACTION_DIM, KEYVAULT_NAME,
                          DB_TOKEN_SECRET, CHANCE_OF_NO_ACTION,
                          GOLD_STORAGE, STATE_MAX_LEN, DEPLOY_MODEL)

from agent.agent_model.databricks_connect import get_state
from agent.state_encoder.model import AttentionEncoderDecoder


def init():
    global ds, agent, encoder, db_token, agent_ckpt, credential, agent_cold_start

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    encoder_path = os.path.join(model_dir, 'INPUT_model_path', 'encoder-ckpt', 'last.ckpt')
    agent_path = os.path.join(model_dir, 'INPUT_model_path', 'agent-ckpt')

    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=f'https://{KEYVAULT_NAME}.vault.azure.net/', credential=credential)

    db_token = secret_client.get_secret(DB_TOKEN_SECRET).value

    agent = DQN(config=AGENT_CONFIG)

    if COLD_START_AGENT_CONFIG:
        agent_cold_start = DQN(config=COLD_START_AGENT_CONFIG)
    else:
        agent_cold_start = None

    if not COLD_START:
        agent.restore(agent_path)

    encoder = AttentionEncoderDecoder.load_from_checkpoint(encoder_path, device='cpu')
    encoder.eval()


def run(raw_data):
    msg_json = json.loads(raw_data)
    orig_app_name = msg_json['app'].lower()

    # this is only for testing purposes.
    priority = msg_json['app_data'].get('priorityaction')
    if '-test' in msg_json['app'].lower() and isinstance(priority, int):
        response = {'actions': [msg_json['app_data']['eligibilityflags'][priority]]}
        return response

    try:
        app_actions = ACTION_MAP[orig_app_name]
        app_name = orig_app_name
    except KeyError:
        app_name = orig_app_name.replace('-test', '')
        app_actions = ACTION_MAP[app_name]

    is_cold_start = COLD_START == app_name
    app_action_map = {**ACTION_MAP['common'], **app_actions}
    app_data = APP_DATA_MAP

    #TODO we need to handle multiple application payload parsing in a more scalable way.
    if app_name == 'ivr':
        sessionid = msg_json['app_data']['ivrcallid']
        raw_id = msg_json['app_data']['cagm']
        raw_intent = msg_json['app_data']['callintent']
        eligibility = list(set(['none'] + msg_json['app_data']['eligibilityflags']))
        action_mask = get_action_mask(app_action_map, eligibility, is_cold_start=is_cold_start, from_string=True)
        info = {'callintent': raw_intent}
    elif app_name == 'spec-pat':
        sessionid = msg_json['app_data']['sessionid']
        raw_id = msg_json['app_data']['cagm']
        raw_intent = msg_json['app_data']['digitalintent']
        eligibility = list(set(['none'] + msg_json['app_data']['eligibilityflags']))
        action_mask = get_action_mask(app_action_map, eligibility,
                                      from_string=True, is_cold_start=is_cold_start)
        info = {'digitalintent': raw_intent, 'patientid': msg_json['app_data'].get('patientid')}
    else:
        raise ValueError(f'app with the name: {app_name} is invalid...')

    tz = pytz.timezone('US/Central')
    time = datetime.now(tz)
    dt_string = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        # balance out the fact that "none" action is always available so is chosen more often.
        if sum(action_mask) > 1 and np.random.uniform() > CHANCE_OF_NO_ACTION:
            action_mask[0] = 0

        raw_state, raw_str, eid = get_state(raw_id, db_token, app_data)
        print('state ', raw_state)
        if len(raw_state) >= STATE_MAX_LEN:
            raw_state = raw_state[-(STATE_MAX_LEN - 1):]
        elif len(raw_state) == 0:
            raw_state = torch.tensor([0])

        with torch.inference_mode():
            obs = encoder.encode(raw_state).squeeze().numpy()

        state = {'mask': np.array(action_mask), 'real_obs': obs}

        if is_cold_start:
            app_action_dim = COLD_START_AGENT_CONFIG['model']['custom_model_config']['action_dim']
            action_dict = agent_cold_start.compute_single_action(state, full_fetch=True)
        else:
            app_action_dim = ACTION_DIM
            action_dict = agent.compute_single_action(state, full_fetch=True)

        action_prob = float(action_dict[2]['action_prob'])
        action_logp = float(action_dict[2]['action_logp'])
        logits = action_dict[2]['action_dist_inputs']
        q_values = action_dict[2]['q_values']
        action = int(action_dict[0])
        print('best action ', action)

        print('allowed actions ', action_mask)
        probs = make_probs(logits)

        if TOPN_ACTIONS and sum(action_mask) > TOPN_ACTIONS:
            action = topn_actions(action, logits, TOPN_ACTIONS - 1, app_action_dim)
            action_friendly = [app_action_map[a] for a in action]
        elif TOPN_ACTIONS and sum(action_mask) <= TOPN_ACTIONS:
            action = topn_actions(action, logits, sum(action_mask) - 1, app_action_dim)
            action_friendly = [app_action_map[a] for a in action]
        else:
            action_friendly = [app_action_map[action]]
            action = [action]

        print('final action ', action, action_friendly)

        data_out = dict(
            app=orig_app_name,
            implementation=DEPLOY_MODEL,
            id=raw_id,
            id_type=app_data['id'],
            enterprise_id=eid,
            time=dt_string,
            t=None,
            eps_id=sessionid,
            agent_index=0,
            obs=obs.tolist(),
            actions=action,
            action_prob=action_prob,
            action_logp=action_logp,
            logits=logits.tolist(),
            q_values=q_values.tolist(),
            probs=probs.tolist(),
            mask=action_mask,
            rewards=None,
            prev_actions=None,
            prev_rewards=None,
            dones=None,
            infos=info,
            new_obs=None,
            raw_state=np.array(raw_state).tolist(),
            raw_state_str=np.append(np.array(raw_str), np.array([raw_intent])).tolist(),
            raw_action_str=action_friendly,
        )
        save_path = 'agent_raw_data'
    except Exception as e:
        data_out = {
            'error': str(e),
            'input': msg_json,
            'time': dt_string
        }
        save_path = 'agent_error_logs'
        action_friendly = [app_action_map[0]]

    try:
        blob_client = BlobServiceClient(credential=credential,
                                        account_url=f'https://{GOLD_STORAGE}.blob.core.windows.net/')
        byte_stream = io.BytesIO(json.dumps(data_out).encode('utf-8'))
        blob = blob_client.get_blob_client('eureka-gold', os.path.join('personalization',
                                                                       save_path,
                                                                       f'{sessionid}_{dt_string}.json'.replace(':', ''))
                                           )
        blob.upload_blob(byte_stream, overwrite=True)
    except Exception as e:
        print(f'error occurred in file upload \n {e}', file=sys.stdout)

    response = {'actions': action_friendly}
    return response


def make_probs(logits, mask_action=None):
    l = logits
    if mask_action:
        l[mask_action] = -1000
    probs = F.softmax(torch.tensor(l), dim=0).numpy()
    probs /= probs.sum()
    return probs


def topn_actions(action, logits, n, action_dim):
    update_probs = make_probs(logits, action)
    action_topn = np.random.choice(list(range(action_dim)), n, p=update_probs, replace=False).tolist()
    action = [action] + action_topn
    return action


def get_action_mask(app_action_map, eligibility, from_string=False, is_cold_start=False):
    if is_cold_start:
        all_actions = sorted([ik for k, v in ACTION_MAP.items() for ik in v.keys()])
    else:
        all_actions = sorted(app_action_map.keys())

    if from_string:
        assert set(eligibility).issubset(app_action_map.values()), 'one or more eligible actions do not match config'
        eligibility = [1 if a in eligibility else 0 for a in app_action_map.values()]

    action_mask = [a for m, a in zip(eligibility, app_action_map) if m == 1]
    valid_actions = [1 if i in app_action_map.keys() and i in action_mask else 0 for i in all_actions]
    return valid_actions


