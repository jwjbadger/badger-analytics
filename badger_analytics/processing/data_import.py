import numpy as np
import requests

def local_import(path):
    try:
        raw = open(path, 'r')
        return np.array(eval(raw.read()))
    finally:
        raw.close()

def remote_import(path):
    raw = requests.get(path)
    return np.array(eval(raw.text))
