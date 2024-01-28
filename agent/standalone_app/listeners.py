import socket
import threading
from episode_handler import handle_request


class IVRSocketListener:
    def __init__(self, host, port, model_addr, sample_batch_path):
        self.host = host
        self.port = port
        self.model_addr = model_addr
        self.sample_batch_path = sample_batch_path

    def run(self):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(1)

        # TODO error handling on timeouts/disconnects
        all_threads = []
        try:
            while True:
                print("Waiting for client")
                conn, addr = s.accept()
                print("Client:", addr)

                t = threading.Thread(target=handle_request, args=(conn, addr, self.model_addr, self.sample_batch_path))
                t.start()

                all_threads.append(t)
        except KeyboardInterrupt:
            print("Stopped by Ctrl+C")
        finally:
            if s:
                s.close()
            for t in all_threads:
                t.join()