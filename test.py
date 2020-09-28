import numpy as np

p = np.load("p.npy")
q = np.load("ab.npy")

print(np.sum(p * (np.log(p) - np.log(q))))