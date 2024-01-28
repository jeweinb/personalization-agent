
import os
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from agent.config import AML_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, help="path to model data")
args = parser.parse_args()

credential = DefaultAzureCredential()
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])

run_model = Model(
    path=args.model_path,
    name='pso-model',
    description='all models for pso are kept in a single model in azureml',
    type='custom_model'
)
ml_client.models.create_or_update(run_model)