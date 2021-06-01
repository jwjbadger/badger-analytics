import random
import numpy as np

# generates theta matrix 
def generate(n):
    result = np.array([])
    
    if n < 1:
        return result
    
    for i in range(1,n):
        np.append(result, random.random() * 10)
