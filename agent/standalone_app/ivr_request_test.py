import socket
import json
from agent.config import HOST, PORT

s = socket.socket()
s.connect((HOST, PORT))

print("Connected to the server")

examples = ['29192494', '12595942']

message = json.dumps({"cagm": "1234566", "done": "False"})
message = message.encode()
s.send(message)

message = s.recv(1024)
message = message.decode()
msg_json = json.loads(message)
print('recv action:', msg_json)

message = json.dumps({"cagm": "1234566", "done": "False"})
message = message.encode()
s.send(message)

message = s.recv(1024)
message = message.decode()
msg_json = json.loads(message)
print('recv action:', msg_json)

message = json.dumps({"cagm": "1234566", "done": "True"})
message = message.encode()
s.send(message)

message = s.recv(1024)
message = message.decode()
msg_json = json.loads(message)
print('recv:', msg_json)
