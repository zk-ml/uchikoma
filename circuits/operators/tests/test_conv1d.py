import os, sys, json
import numpy as np

a = np.ones((4, 10))
b = np.ones((3, 10, 8))
c = np.ones((8))
a = a*8
b = b*2
c = c*3
al = a.tolist()
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        al[i][j] = str(int(a[i][j]))

bl = b.tolist()
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        for k in range(b.shape[2]):
            bl[i][j][k] = str(int(b[i][j][k]))

cl = c.tolist()
for i in range(c.shape[0]):
    cl[i] = str(int(c[i]))

json_data = json.dumps({"in":al, "weights":bl, "bias":cl})
print(json_data)
