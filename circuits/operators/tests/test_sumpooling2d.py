import os, sys, json
import numpy as np

a = np.ones((32, 32, 3))
a = a*8
al = a.tolist()
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            al[i][j][k] = str(int(a[i][j][k]))

json_data = json.dumps({"in":al})
print(json_data)
