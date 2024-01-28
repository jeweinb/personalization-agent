import os, sys, uuid, shutil, argparse, json
from datetime import datetime

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (Model, KubernetesOnlineDeployment, IdentityConfiguration, BatchDeployment, BatchEndpoint,
                                  KubernetesOnlineEndpoint, KubernetesCompute, ResourceRequirementsSettings,
                                  TargetUtilizationScaleSettings, OnlineRequestSettings, ResourceSettings,
                                  CodeConfiguration, Environment)
from azure.storage.blob import BlobServiceClient
from agent.config import AML_CONFIG, SERVICE_PRINCIPAL, AML_DEPLOY_ENV, GOLD_STORAGE


for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

credential = DefaultAzureCredential()
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])

print(os.getcwd())
time = datetime.now()
dt_string = time.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

DEBUG = False
DEPLOY_TEST = True
MIN_MAX_NODES = (1, 1)  # test is (10, 20) prod is (20, 60)

endpoint = BatchEndpoint(
    name='batch-agent-endpoint' if not DEPLOY_TEST else 'batch-agent-endpoint-test',
    compute='aks-v2',
    description="personalized smart offering model",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

model = ml_client.models.get(name='combined-agent-model', label='latest')

deployment = KubernetesOnlineDeployment(name='agent-batch-deployment',
                                        endpoint_name=endpoint.name,
                                        model=model,
                                        environment=AML_DEPLOY_ENV,
                                        instance_count=3,
                                        code_configuration=CodeConfiguration(
                                            code='agent', scoring_script='agent_model/score.py'
                                        ),
                                        app_insights_enabled=True,
                                        request_settings=OnlineRequestSettings(max_queue_wait_ms=60000),
                                        scale_settings=TargetUtilizationScaleSettings(
                                            max_instances=MIN_MAX_NODES[1],
                                            min_instances=MIN_MAX_NODES[0],
                                            target_utilization_percentage=70,
                                        ),
                                        resources=ResourceRequirementsSettings(
                                            requests=ResourceSettings(cpu='100m', memory='0.5Gi'),
                                        ))
ml_client.begin_create_or_update(deployment, local=DEBUG).result()
access_key = ml_client.online_endpoints.get_keys(name=endpoint.name).primary_key
print(f'\n bearer token {access_key}')
