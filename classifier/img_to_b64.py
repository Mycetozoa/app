import base64

def encode(path):
    with open(path, 'rb') as pic:
        data = base64.b64encode(pic.read())
        return data.decode('utf-8')
