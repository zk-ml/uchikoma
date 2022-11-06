import os, sys, json
import numpy as np

a = np.ones((4, 10, 8))
b = np.ones((3, 4, 5, 5))
a = a*8
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
            for m in range(b.shape[3]):
                bl[i][j][k][m] = str(int(b[i][j][k][m]))

json_data = json.dumps({"in":al, "weights":bl})
print(json_data)
