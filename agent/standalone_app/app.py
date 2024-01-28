
from ray import serve
from agent_model import ServeAgentModel
from agent.standalone_app.listeners import IVRSocketListener
from agent.config import HOST, PORT, CHECKPOINT_PATH, SAMPLE_BATCH_PATH, MODEL_ADDRESS


serve.start()
ServeAgentModel.deploy(CHECKPOINT_PATH)

listener = IVRSocketListener(HOST, PORT, MODEL_ADDRESS, SAMPLE_BATCH_PATH)
listener.run()
