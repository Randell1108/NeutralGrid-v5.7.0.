import socket
s=socket.socket()
s.settimeout(0.3)
try:
    s.connect(('127.0.0.1', 9222))
    print('open')
except Exception as e:
    print('closed', e)
finally:
    s.close()
