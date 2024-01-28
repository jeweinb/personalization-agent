from ray.rllib.algorithms.dqn import DQN
from ray import serve
from starlette.requests import Request
from agent.config import ENV_CONFIG


def train_agent_model(checkpoint_path, sample_batch_path=None):
    trainer = DQN(config={
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "input": sample_batch_path,
            **ENV_CONFIG,
        },
    )
    trainer.train()
    trainer.save(checkpoint_path)


@serve.deployment(route_prefix="/ivr-agent")
class ServeAgentModel:
    def __init__(self, checkpoint_path) -> None:
        self.trainer = DQN(config={
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            **ENV_CONFIG,
            },
        )
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_single_action(obs, full_fetch=True)
        return {"action": int(action[0]),
                "action_prob": float(action[2]['action_prob']),
                "action_logp": float(action[2]['action_logp'])}
# query the agent model and get action
# model_response = requests.post(MODEL_ADDRESS, json={"observation": state.tolist()})
# action = model_response.json()


if __name__ == '__main__':
    CHECKPOINT_PATH = '../model_checkpoints/mac_agent'
    SAMPLE_BATCH_PATH = '../agent_sample_batches'
    train_agent_model(CHECKPOINT_PATH, SAMPLE_BATCH_PATH)
