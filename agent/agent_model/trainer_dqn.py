
import os, io, json
import argparse

from ray import tune
from ray.tune import TuneConfig
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.offline.estimators import (
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.tune import Tuner
from ray import air
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.storage.blob import BlobServiceClient

from agent.agent_model.mlflow_callback import CustomMLflowLogger
from agent.config import ENV_CONFIG, SERVICE_PRINCIPAL, AML_CONFIG, GOLD_STORAGE
from agent.agent_model.trainer_utils import (MLFLOW_URL, BATCH_PATH, validate_model,
                                             gather_sample_batches)

parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, help="path to train data")
parser.add_argument("--output-model", type=str, help="path to save model")
args = parser.parse_args()

output_path = os.path.join(args.output_model, 'agent-ckpt')

for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

credential = DefaultAzureCredential()
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])

explore_config = (
    DQNConfig()
    .rollouts(rollout_fragment_length=tune.randint(1, 20),
              batch_mode='complete_episodes'
    )
    .resources(num_gpus=0)
    .debugging(seed=67)
    .environment(**ENV_CONFIG)
    .training(
        model={'custom_model': 'maskable_dqn'},
        hiddens=[],
        dueling=False,
        gamma=0.99,
        v_max=2,
        v_min=2,
        grad_clip=tune.uniform(0.01, 100),
        lr=tune.uniform(0.00001, 0.001),
        n_step=tune.randint(1, 10),
        train_batch_size=tune.randint(16, 128),
        adam_epsilon=tune.uniform(0.00001, 0.01),
        target_network_update_freq=tune.grid_search([0, 1, 100, 500]),
        replay_buffer_config={
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": tune.grid_search([100, 1000, 5000])}
    )
    .framework("torch")
    .offline_data(input_=os.path.join(BATCH_PATH, 'train'))
    .exploration(explore=True,
                 exploration_config={
                    'type': 'EpsilonGreedy',
                    'initial_epsilon': 0.8,
                    'final_epsilon': 0.1,
                    'epsilon_timesteps': 1000}
    )
    .reporting(min_time_s_per_iteration=20, min_train_timesteps_per_iteration=1000)
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=10,
        evaluation_num_workers=0,
        evaluation_parallel_to_training=False,
        evaluation_duration_unit="episodes",
        evaluation_config={"input": os.path.join(BATCH_PATH, 'eval')},
        off_policy_estimation_methods={
            "is": {"type": ImportanceSampling},
            "wis": {"type": WeightedImportanceSampling},
            "dr_fqe": {
                "type": DoublyRobust,
                "q_model_config": {"type": FQETorchModel, "polyak_coef": 0.05},
            },
        },
    )
)

config = (
    DQNConfig()
    .rollouts(rollout_fragment_length=12,
              batch_mode='complete_episodes'
    )
    .resources(num_gpus=0)
    .debugging(seed=67)
    .environment(**ENV_CONFIG)
    .training(
        model={'custom_model': 'maskable_dqn'},
        hiddens=[],
        dueling=False,
        gamma=0.99,
        v_max=2,
        v_min=2,
        grad_clip=82,
        lr=0.000785,
        n_step=6,
        train_batch_size=64,
        adam_epsilon=0.00517,
        target_network_update_freq=100,
        replay_buffer_config={
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 1000}
    )
    .framework("torch")
    .offline_data(input_=os.path.join(BATCH_PATH, 'train'))
    .exploration(explore=True,
                 exploration_config={
                    'type': 'EpsilonGreedy',
                    'initial_epsilon': 0.8,
                    'final_epsilon': 0.1,
                    'epsilon_timesteps': 1000}
    )
    .reporting(min_time_s_per_iteration=20, min_train_timesteps_per_iteration=1000)
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=10,
        evaluation_num_workers=0,
        evaluation_parallel_to_training=False,
        evaluation_duration_unit="episodes",
        evaluation_config={"input": os.path.join(BATCH_PATH, 'eval')},
        off_policy_estimation_methods={
            "is": {"type": ImportanceSampling},
            "wis": {"type": WeightedImportanceSampling},
            "dr_fqe": {
                "type": DoublyRobust,
                "q_model_config": {"type": FQETorchModel, "polyak_coef": 0.05},
            },
        },
    )
)

if __name__ == '__main__':
    gather_sample_batches(args.train_data, train_size=0.7)

    mflow = CustomMLflowLogger(tracking_uri=MLFLOW_URL, experiment_name='train-dqn')
    stop = {"training_iteration": 20}
    stopper = TrialPlateauStopper(metric="evaluation/off_policy_estimator/is/v_gain",
                                  std=0.1, num_results=4, grace_period=4)

    t = Tuner(DQN,
              param_space=explore_config.to_dict(),
              run_config=air.RunConfig(stop=stop, callbacks=[mflow]),
              tune_config=TuneConfig(
                  mode="max",
                  metric='evaluation/off_policy_estimator/is/v_gain',
                  num_samples=10,
                  max_concurrent_trials=40,
                  # search_alg=OptunaSearch(),
                ),
              )
    results = t.fit()

    best_model = results.get_best_result(metric='evaluation/off_policy_estimator/is/v_gain', mode='max')
    best_model.config['off_policy_estimation_methods'] = config.off_policy_estimation_methods
    best_model.checkpoint.to_directory(output_path)

    all_records, agg_df = validate_model(args.train_data,
                                         DQN(config=best_model.config),
                                         best_model.checkpoint._local_path
                                         )
    agg_df.to_csv(os.path.join(args.output_model, 'action_dist.csv'), index=False)

    run_model = Model(
        path=output_path,
        name='pso-model',
        description='all models for pso are kept in a single model in azureml',
        type='custom_model'
    )
    ml_client.models.create_or_update(run_model)
