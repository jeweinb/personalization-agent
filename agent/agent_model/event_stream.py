import os, io
import asyncio
from azure.ai.ml import MLClient

from azure.eventhub import EventData
from azure.eventhub import EventHubProducerClient
from azure.identity import DefaultAzureCredential
from agent.config import SERVICE_PRINCIPAL


for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

FULLY_QUALIFIED_NAMESPACE = ''
EVENT_HUB_NAME = "test-stream"
CONSUMER_GROUP = '$Default'

credential = DefaultAzureCredential()


def run():
    producer = EventHubProducerClient(eventhub_name=EVENT_HUB_NAME,
                                      credential=credential,
                                      fully_qualified_namespace=FULLY_QUALIFIED_NAMESPACE)
    with producer:
        event_data_batch = producer.create_batch()
        event_data_batch.add(EventData('test data'))
        producer.send_batch(event_data_batch)


run()
