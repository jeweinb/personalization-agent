
import os, json, sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

import torch
torch.cuda.empty_cache()
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

sys.path.append('./agent')
from agent.state_encoder.model import AttentionEncoderDecoder
from agent.state_encoder.data_loaders import StateDataModule
from agent.config import (AML_CONFIG, SERVICE_PRINCIPAL, STATE_DIM,
                          STATE_MAX_LEN, ENCODER_RESUME_TRAINING, DEPLOY_MODEL,
                          NUM_NODES, NUM_DEVICES_PER_NODE, ENCODER_BATCH_SIZE, GRAD_ACCUM_BATCHES)


time = datetime.now()
dt_string = time.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, help="path to train data")
parser.add_argument("--token-data", type=str, help="feature space size")
parser.add_argument("--output-model", type=str, help="path to save model")
args = parser.parse_args()

for k, v in SERVICE_PRINCIPAL.items():
    os.environ[k] = v

credential = DefaultAzureCredential()
ml_client = MLClient(credential,
                     AML_CONFIG['subscription_id'],
                     AML_CONFIG['resource_group'],
                     AML_CONFIG['workspace_name'])

dataset = pd.read_parquet(args.train_data)

data_module = StateDataModule(dataset, state_col='state_tokenized',
                              split_on_col='enterprise_id', split_sizes=[0.8, 0.2],
                              samp_size_per_epoch=None,
                              batch_size=ENCODER_BATCH_SIZE, num_workers=6, train_test_split=True,
                              max_seq_len=STATE_MAX_LEN, pad_token=0, pad_side='right')

num_tokens = pd.read_parquet(args.token_data)
num_tokens = num_tokens['token'].astype('int').max() + 1

if ENCODER_RESUME_TRAINING:
    ml_client.models.download('state-encoder', version='latest', download_path='.')
    model = AttentionEncoderDecoder.load_from_checkpoint('last.ckpt')
    model.update_vocab(num_tokens)
else:
    model = AttentionEncoderDecoder(vocab_size=num_tokens, emb_dim=64, feature_dim=STATE_DIM,
                                    seq_len=STATE_MAX_LEN, dropout=0.1, num_layers=1, lr=0.0001)

print(model.vocab_size)
print(model)

mlflow_url = os.environ['MLFLOW_TRACKING_URI']
mlf_logger = MLFlowLogger(experiment_name=f'train-encoder-{DEPLOY_MODEL}', tracking_uri=mlflow_url)

callbacks = [ModelCheckpoint(monitor='loss',
                             dirpath=os.path.join(args.output_model, 'encoder-ckpt'),
                             save_last=True,
                             save_top_k=1,
                             mode='min')]
early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=False, mode='min')
callbacks.append(early_stop_callback)

trainer = pl.Trainer(accelerator='auto',
                     devices=NUM_DEVICES_PER_NODE,
                     strategy=pl.strategies.DDPStrategy(timeout=timedelta(seconds=20000),
                                                        process_group_backend='gloo',
                                                        find_unused_parameters=False),
                     num_nodes=NUM_NODES,
                     callbacks=callbacks,
                     logger=mlf_logger,
                     num_sanity_val_steps=0,
                     max_epochs=1,
                     precision=16,
                     reload_dataloaders_every_n_epochs=1,
                     accumulate_grad_batches=GRAD_ACCUM_BATCHES,
                     )
trainer.fit(model, data_module)

if trainer.is_global_zero:
    run_model = Model(
        path=args.output_model,
        name='pso-model',
        description='all models for pso are kept in a single model in azureml',
        type='custom_model'
    )
    ml_client.models.create_or_update(run_model)
