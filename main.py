import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback

def main():
    HOST = socket.gethostname()
    PORT = 10012

    server = CustomSocket(HOST, PORT)
    server.startServer()

    while True:
        # Wait for connection from client
        conn, addr = server.sock.accept()
        print("Clent connected from", addr)

        while True:
            try:
                data = server.recvMsg(conn)
                img = np.frombuffer(data, dtype=np.uint8).reshape(720, 1280, 3)

                # Show image
                cv2.imshow("Server Cam", img)
                if cv2.waitKey(1) == ord('q'):
                    break
            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Connection Closed")
                del res
                break
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()