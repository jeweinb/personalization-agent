from ray.rllib.models import ModelCatalog
from agent.agent_model.model import MaskableDQN
ModelCatalog.register_custom_model(
    "maskable_dqn",
    MaskableDQN
)