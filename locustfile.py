import json, os
import numpy as np
from locust import HttpUser, task
from agent.config import ACTION_MAP, ACTION_DIM, COLD_START
import ssl


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


TOKEN = 'xxx'
ENDPOINT = 'https://xxx.com/api/v1/endpoint/agent-endpoint/score'
HEADERS = {'Content-Type': 'application/json',
           'Authorization': ('Bearer ' + TOKEN),
           "User-Agent": "pso-agent",
           'azureml-model-deployment': 'agent-deployment'}


class TestAgentEndpoint(HttpUser):

    def on_start(self):
        self.client.headers = HEADERS
        self.examples = EXAMPLES

    @task(1)
    def users(self):
        cagm = np.random.choice(self.examples, 1)[0]
        app_action_map = {**ACTION_MAP['common'], **ACTION_MAP['spec-pat']}

        INTENT = np.random.choice([a for k, a in app_action_map.items()], 1)[0]
        ELIGIBILITY_FLAGS = (np.random.rand(len(app_action_map) - 1, ) > .5).astype(int).tolist()

        query = f'''{{"app": "SPEC-PAT-TEST", "app_data": {{
            "cagm": "{cagm}",
            "sessionid": "test-test-test",
            "patientid": "12345",
            "digitalintent": "{INTENT}",
            "eligibilityflags": {ELIGIBILITY_FLAGS}
            }}}}'''

        response = self.client.post(
            url=ENDPOINT,
            data=query,
            verify='xxxx.pem'
        )

        print(response)


EXAMPLES = [
]