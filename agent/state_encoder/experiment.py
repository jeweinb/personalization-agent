
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
    ml_client.jobs.begin_cancel(f'train-encoder-{DEPLOY_MODEL}')
else:
    inputs = {
        "train_data": Input(type='uri_folder', path=f'{DATA_STORE}/personalization_state_train'),
        "token_data": Input(type='uri_folder', path=f'{DATA_STORE}/personalization_state_train_tokens')
    }

    outputs = {
        'output_model': Output(type='uri_folder', mode='rw_mount',
                               path=f'{DATA_STORE}/model_artifacts/{DEPLOY_MODEL}')
    }

    dist_config = {"type": "PyTorch", "process_count_per_instance": NUM_DEVICES_PER_NODE}
    job = command(
        experiment_name=f'train-encoder-{DEPLOY_MODEL}',
        description=f'training encoder model {DEPLOY_MODEL}',
        inputs=inputs,
        outputs=outputs,
        code='.',
        command='''python -m agent.state_encoder.trainer --train-data ${{inputs.train_data}} --token-data ${{inputs.token_data}} --output-model ${{outputs.output_model}}''',
        environment=ml_client.environments.get('custom-train-env-v2', version='9'),
        compute=TRAIN_COMPUTE,
        instance_count=NUM_NODES,
        distribution=dist_config if NUM_NODES > 1 else None,
    )
    job = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(job.name)
