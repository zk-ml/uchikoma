import os, sys, json
import numpy as np

a = np.ones((128))
b = np.ones((128, 128))
a = a*8
b = b*2
al = a.tolist()
for i in range(a.shape[0]):
    al[i] = str(int(a[i]))

bl = b.tolist()
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        bl[i][j] = str(int(b[i][j]))

json_data = json.dumps({"in":al, "weights":bl})
print(json_data)
