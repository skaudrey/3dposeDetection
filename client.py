#!/usr/bin/env python
# encoding: utf-8
'''
@author: Miao Feng
@contact: skaudreymia@gmail.com
@software: PyCharm
@file: client.py
@time: 2020/12/18 17:55
@desc:
'''

import socket

__config__ = {
    'host':'127.0.0.1',
    'port':25001
}

# HOST = '192.168.0.19'
HOST = '127.0.0.1'

PORT = 25001
data = "0,0,0"

client = socket.socket(socket.SOCK_DGRAM)
client.connect((__config__['host'], __config__['port']))  # bind server

try:
    while True:
        msg = input(">>:").strip() # input request from console
        if len(msg) == 0:
            continue

        client.send(msg.encode("utf-8"))
        if msg == 'break':
            break
        data = client.recv(4096)
        print("recv:", data.decode())
finally:
    client.close()