
from azure.ai.ml import MLClient
from azure.ai.ml import MLClient, command, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

from agent.config import (AML_CONFIG, COMPUTES, IMG_ENVS, NUM_NODES)

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'],
                     )

encoder_data_component = command(

)