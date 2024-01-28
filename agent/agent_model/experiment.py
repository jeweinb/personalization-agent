
import os
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential

from agent.config import AML_CONFIG, AML_TRAIN_ENV, SERVICE_PRINCIPAL
from agent.config import TRAIN_COMPUTE, NUM_NODES, NUM_DEVICES_PER_NODE, DEPLOY_MODEL, DATA_STORE


CANCEL = False

for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

credential = DefaultAzureCredential()
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])

if CANCEL:
    ml_client.jobs.begin_cancel(f'train-agent-{DEPLOY_MODEL}')
else:
    inputs = {
        "train_data": Input(type='uri_folder', path=f'{DATA_STORE}/personalization_train_episodes'),
    }

    outputs = {
        'output_model': Output(type='uri_folder', mode='rw_mount',
                               path=f'{DATA_STORE}/model_artifacts/{DEPLOY_MODEL}')
    }

    job = command(
        experiment_name=f'train-agent-{DEPLOY_MODEL}',
        description=f'training agent model {DEPLOY_MODEL}',
        inputs=inputs,
        outputs=outputs,
        code='.',
        command='''python -m agent.agent_model.trainer_dqn --train-data ${{inputs.train_data}} --output-model ${{outputs.output_model}}''',
        environment=ml_client.environments.get('custom-train-env-v2', version='9'),
        compute='lrg-cpu-cluster',
        instance_count=1,
    )
    job = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(job.name)
