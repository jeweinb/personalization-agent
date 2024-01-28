
import os, json
import torch
import torch.nn.functional as F
import numpy as np

import agent.state_encoder.model
import importlib
importlib.reload(agent.state_encoder.model)

from azureml.core import Workspace
from agent.config import (AML_CONFIG, SERVICE_PRINCIPAL, ENCODER_RESUME_TRAINING,
                          STATE_DIM, STATE_MAX_LEN, SECRET)
from azureml.core.authentication import ServicePrincipalAuthentication
from agent.state_encoder.model import AttentionEncoderDecoder
from agent.state_encoder.data_loaders import StateDataset


SAVE_PATH = './model_checkpoints/encoder'

sp = ServicePrincipalAuthentication(tenant_id=SERVICE_PRINCIPAL['tenant_id'],
                                    service_principal_id=SERVICE_PRINCIPAL['client_id'],
                                    service_principal_password=SECRET)

ws = Workspace(**AML_CONFIG, auth=sp)


import importlib
import agent.state_encoder.model
importlib.reload(agent.state_encoder.model)
from agent.state_encoder.model import AttentionEncoderDecoder

example1 = torch.tensor(np.random.randint(0, 15921, size=(1, 100)))
example2 = torch.tensor(np.random.randint(0, 15921, size=(1, 100)))
example = torch.cat([example1, example2]).squeeze(0)


model = AttentionEncoderDecoder(vocab_size=15921 + 1, emb_dim=64, feature_dim=STATE_DIM,
                           seq_len=STATE_MAX_LEN, dropout=0.0, num_layers=1, lr=0.0001)

print(example.size())
r = model(example)
print(r.size())
t = model.encode(example1.squeeze(0))
print(t.size())
