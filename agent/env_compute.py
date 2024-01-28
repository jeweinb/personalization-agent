
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, AmlCompute, IdentityConfiguration
from azure.identity import DefaultAzureCredential

from agent.config import (AML_CONFIG, COMPUTES, IMG_ENVS, NUM_NODES)

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'],
                     )

for c in COMPUTES:
    compute = AmlCompute(name=c['name'], size=c['size'],
                         min_instances=0, max_instances=c['max'],
                         tier='Dedicated',
                         identity=IdentityConfiguration(type='system_assigned'),
                         # network_settings=NetworkSettings(vnet_name=VNET_NAME, subnet=TRAIN_SUBNET_NAME)
                         )
    ml_client.compute.begin_create_or_update(compute)

### TODO this is used to register our custom docker images from jenkins. We are using base azureml envs instead
for e in IMG_ENVS:
    env = Environment(
        name=e['name'],
        image=e['base'],
        version='latest',
    )
    ml_client.environments.create_or_update(env)

# env = Environment(image='mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
#                   conda_file='conda_env.yaml',
#                   name='custom-inference-env-v2')
# ml_client.environments.create_or_update(env)

env = Environment(image='mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04',
                  conda_file='conda_env.yaml',
                  name='custom-train-env-v2'
                  )
ml_client.environments.create_or_update(env)
