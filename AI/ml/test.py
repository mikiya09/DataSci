

import numpy as np 


nArray = np.ones((4, 7))
nArray = np.array(
         [[2, 3, 5, 7, 1, 13, 1],
          [1, 1, 6, 8, 2, 3, 0],
          [1, 2, 1, 3, 5, 7, 1],
          [1, 5, 1, 7, 1, 9, 0]])

def meanFeature(data, classVal):
    outMeans = np.zeros(data.shape[1])

    for i in range(data.shape[1]):
        inds = np.where(data[:, -1] == classVal)[0]
        indsx = np.where(data[:, -1] == classVal)

        print(indsx)
        outMeans[i] = data[inds, i].mean()

        print(data[inds, i])
        print(f"inds: {inds} | inds type: {type(inds)} | i: {i}")
        print()

    return outMeans

print(nArray)
print(type(nArray))
print(nArray.shape[1])
print(np.zeros(nArray.shape[1]))
print('--------------------')
print(meanFeature(nArray, 1))
