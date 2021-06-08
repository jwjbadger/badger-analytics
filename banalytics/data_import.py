import numpy as np
import requests

def local_import(path):
    """
    local_import(path)

    Import data from a file on your computer.

    Parameters
    ----------
    path : string
        the path to the file to open
    
    Returns
    -------
    output : numpy array
        an array with the read data
    """
    try:
        raw = open(path, 'r')
        return np.array(eval(raw.read()))
    finally:
        raw.close()

def remote_import(path):
    """
    remote_import(path)

    Import a file from a URL.

    Parameters
    ----------
    path : string
        the URL to the file to get
    
    Returns
    -------
    output : numpy array
        an array with the read data
    """
    raw = requests.get(path)
    return np.array(eval(raw.text))
