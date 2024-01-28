
import sys
from gym.spaces import Box
import torch
import torch.nn as nn
import numpy as np
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX


#TODO update this model to only do simple action masking (no embedding)
#https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
class MaskableDQN(DQNTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, action_dim,
                 true_obs_shape, action_embed_size=4, **kw):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )
        self.action_embed_model = TorchFC(
            Box(-np.inf, np.inf, shape=true_obs_shape),
            action_space,
            action_embed_size,
            model_config,
            name=name + "_action_embed",
        )

        self.embed_dim = action_embed_size
        self.avail_actions = torch.nn.Parameter(torch.zeros((action_dim, self.embed_dim)))
        self.embeddings = nn.Embedding(action_dim, self.embed_dim)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict['obs']['mask']
        real_obs = input_dict['obs']['real_obs']
        avail_actions = self.avail_actions.clone().expand(action_mask.size(0), -1, -1)

        for i, m in enumerate(action_mask):
            idx = torch.nonzero(m, as_tuple=True)[0].long()
            avail_embeds = self.embeddings(idx)
            avail_actions[i, idx] = avail_embeds

        action_embed, _ = self.action_embed_model({'obs': real_obs})
        intent_vector = torch.unsqueeze(action_embed, 1)
        try:
            action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        except:
            print(f'action_mask size: {action_mask.size()}, real_obs size: {real_obs.size()}', file=sys.stdout)
            print(f'avail size: {avail_actions.size()}, intent size: {intent_vector.size()}', file=sys.stdout)
            print(f'avail size: {action_embed.size()}', file=sys.stdout)

            action_logits = None

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
    