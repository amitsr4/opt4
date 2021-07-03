import copy

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA



def f(x, mu):
    x1 = x[0]
    x2 = x[1]
    return (x1+x2)**2 - 10*(x1+x2)+ mu/2 *((3*x1+x2-6)**2)+ mu/2 *(max(x1**2 + x2**2 -5, 0)**2) + mu/2* (max(-1*x1, 0)**2)

def gradf(x, mu):
    x1 = x[0]
    x2 = x[1]
    ineq1_x1 = mu * 2*x1*(x1**2 +x2**2 -5) if x1**2 +x2**2 -5 >0 else 0
    ineq1_x2 = mu * 2*x2*(x1**2 +x2**2 -5) if (x1**2 +x2**2 -5 >0) else 0
    ineq2_x1 = 1*mu * x1  if -1 * x1>0 else 0
    ineq2_x2 = 0
    gradx1 = 2*x1+2*x2-10+3*mu *(3*x1 +x2 -6) + ineq1_x1 + ineq2_x1
    gradx2 = 2*x1+2*x2-10+mu *(3*x1 +x2 -6) + ineq1_x2 + ineq2_x2
    return np.asarray([gradx1, gradx2])

def Armijio(x, gradx, d, a, b, c, maxiter, mu):
    for i in range(maxiter):
        obj_a = f(x + a*d, mu)
        if obj_a < (f(x, mu) + c*a*np.inner(gradx,d)):
            return a
        else:
            a = b*a
    return a


def gradient_descent( x, maxIter, a0, beta, c, epsilon, mu, f_val):
    f_val.append(f(x, mu))
    # x1_k = [x[0]]
    # x2_k = [x[1]]
    x_k = [x]
    for i in range(maxIter):
        xprev = x
        grad = gradf(x, mu)
        d = -grad
        alpha = Armijio(x, grad, d, a0, beta, c, 100, mu)
        x = x + alpha * d
        # x1_k.append(x[0])
        # x2_k.append(x[1])
        x_k.append(x)
        f_val.append(f(x, mu))
        if i >1 and LA.norm(x - xprev) / LA.norm(xprev) < epsilon:
            break
    return x, f_val, x_k

alpha = 0.25
beta = 0.5
c = 1e-4
epsilon = 0.01

fxStar = np.asarray([1.242, 1.72])
mu = [0.01,0.1,1,10,100]
x = np.asarray([0,0])
f_val =[]
for i in range(5):
    x, f_val, x_k = gradient_descent(x, 50, alpha, beta, c, epsilon, mu[i], f_val)
    print(x)



plt.figure()
plt.plot(np.arange(len(f_val)),f_val)
plt.xlabel("iterations")
plt.ylabel(r'$f(x_k)$')
plt.title("function values")
plt.show()

# plt.figure()
# plt.plot(np.arange(len(f_norms)),f_norms)
# plt.xlabel("iterations")
# plt.show()


