import requests, ssl, os
import numpy as np
from agent.config import ACTION_MAP, ACTION_DIM, COLD_START


TOKEN = ''
ENDPOINT = 'https://xxx.com/api/v1/endpoint/agent-endpoint/score'
HEADERS = {'Content-Type': 'application/json',
           'Authorization': ('Bearer ' + TOKEN),
           "User-Agent": "pso-agent",
           'azureml-model-deployment': 'agent-deployment'}

EXAMPLES = [

]

for id in EXAMPLES:
    INTENT = np.random.choice([a for k, a in ACTION_MAP.items()], 1)[0]
    app_action_map = {**ACTION_MAP['common'], **ACTION_MAP['spec-pat']}
    ELIGIBILITY_FLAGS = (np.random.rand(len(app_action_map) - 1,) > .5).astype(int).tolist()

    query = f'''{{"app": "SPEC-PAT-TEST", "app_data": {{
        "cagm": "{id}",
        "sessionid": "test-test-test",
        "patientid": "12345",
        "digitalintent": "{INTENT}",
        "eligibilityflags": {ELIGIBILITY_FLAGS}
        }}}}'''

    response = requests.post(ENDPOINT, query, headers=HEADERS, verify='../../xxxcabundle.pem')
    print(response.status_code)
    print(response.text)
