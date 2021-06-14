import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import copy



def min_xi(x, H, g, i, a , b):
    sum=0
    for j in range(len(x)):
        if i != j:
            sum += (H[j][i] * x[j])
    sum = -1 * (sum- g[i])/ (H[i][i])
    if sum <a[i]:
        return a[i]
    elif sum >b[i]:
        return b[i]
    return sum

def coordinateDescent(x, H, g,a,b, maxIter, epsilon):
    x_k = [copy.copy(x)]
    for k in range(maxIter):
        for i in range(len(x)):
            x[i]  = min_xi(x,H,g,i,a,b)
        x_k.append(copy.copy(x))
        if LA.norm(x - x_k[k]) / LA.norm(x_k[k]) < epsilon:
            break

    return x, x_k


H = 6*np.eye(5) + np.full((5,5),-1)
g = np.asarray([18,6,-12,-6,18])
a = np.zeros(5)
b = np.ones(5) *5
x0 = np.random.random(5)


x,x_k = coordinateDescent(x0, H, g, a, b, 10000,0.0001)
print(x)