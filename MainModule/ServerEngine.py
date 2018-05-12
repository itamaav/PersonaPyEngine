

import socket

soc = socket.socket()
host = "localhost"
port = 2004
soc.bind((host, port))
soc.listen(5)

while True:
    conn, addr = soc.accept()
    print("Got connection from",addr)
    length_of_message = int.from_bytes(conn.recv(2), byteorder='big')
    msg = conn.recv(length_of_message).decode("UTF-8")
    print(msg)
    print(length_of_message)

    # Note the corrected indentation below
    if "Hello"in msg:
        message_to_send = "bye".encode("UTF-8")
        conn.send(len(message_to_send).to_bytes(2, byteorder='big'))
        conn.send(message_to_send)
    else:
        print("no message")