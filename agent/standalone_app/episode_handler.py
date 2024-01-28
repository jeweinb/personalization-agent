
import json
import numpy as np
import requests
import uuid
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from agent.config import ENV_CONFIG


def handle_request(conn, addr, model_addr, sample_batch_path):
    print("[thread] starting")

    obs_prep = get_preprocessor(ENV_CONFIG['observation_space'])(ENV_CONFIG['observation_space'])
    eid = uuid.uuid4()

    msg_json = receive_msg(conn)
    done = eval(msg_json['done'])
    print("[thread] client:", addr, 'recv:', msg_json)

    samples = []
    prev_action = np.zeros_like(ENV_CONFIG['action_space'].sample())
    prev_reward = 0
    steps = 0
    while not done:
        # TODO use cagm and query state
        state = np.random.rand(5)

        # query the agent model and get action
        model_response = requests.get(model_addr, json={"observation": state.tolist()})
        action = model_response.json()
        action_prob = action['action_prob']
        action_logp = action['action_logp']

        # send the suggested action back to the ivr client
        send_msg(conn, action['action'])

        # get next state info and determine if episode should be completed
        msg_json = receive_msg(conn)
        done = eval(msg_json['done'])
        print('recv: ', msg_json)

        # TODO get most recent state and predict reward
        new_state = np.random.rand(5)
        reward = np.random.binomial(1, .5)

        samples.append(dict(
            t=steps,
            eps_id=eid,
            agent_index=0,
            obs=obs_prep.transform(state),
            actions=action['action'],
            action_prob=action_prob,
            action_logp=action_logp,
            rewards=reward,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=int(done),
            infos={},
            new_obs=obs_prep.transform(new_state)
        ))
        steps += 1
        prev_action = action['action']
        prev_reward = reward

    end_msg = {"cagm": msg_json['cagm'], "done": "true"}
    send_msg(conn, end_msg)
    print("[thread] client:", addr, 'send:', end_msg)

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(sample_batch_path)
    for s in samples:
        batch_builder.add_values(**s)
    writer.write(batch_builder.build_and_reset())

    conn.close()


def send_msg(conn, msg):
    msg_json = json.dumps(msg)
    msg_json = msg_json.encode()
    conn.send(msg_json)


def receive_msg(conn):
    message = conn.recv(1024)
    message = message.decode()
    msg_json = json.loads(message)
    return msg_json
