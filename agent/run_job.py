
import os
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

from agent.config import AML_CONFIG, SERVICE_PRINCIPAL, DATA_STORE, DEPLOY_MODEL

for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

credential = DefaultAzureCredential()
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])

# ENTRY POINT
# generic script for running scripts on azureml compute resources instead of local.
# just edit the command entry to run any script you want. You can add inputs/ouput paths as well.

model_path = os.path.join(DATA_STORE, 'model_artifacts', DEPLOY_MODEL)
inputs = {'model_path': Input(type='uri_folder', path=model_path)}

job = command(
    experiment_name='register_model',
    description='register pso models for prod deploy',
    inputs=inputs,
    code='.',
    command='''python -m agent.register_model --model-path ${{inputs.model_path}}''',
    environment=ml_client.environments.get('custom-train-env-v2', version='9'),
    compute='training',
    instance_count=1,
)
job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(job.name)
