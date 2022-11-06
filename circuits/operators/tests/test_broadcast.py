import os, sys, json
import numpy as np

a = np.ones((256, 3, 3))
b = np.ones((256, 1, 1))
b = b*2
al = a.tolist()
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            al[i][j][k] = str(int(a[i][j][k]))

bl = b.tolist()
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        for k in range(b.shape[2]):
            bl[i][j][k] = str(int(b[i][j][k]))

json_data = json.dumps({"A":al, "B":bl})
print(json_data)
