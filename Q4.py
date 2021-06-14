import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import scipy.sparse as sp



def f(A, C, x, b, lam):
    return LA.norm((A@C@x) - b) **2 +lam * (np.ones(len(x)) @ x)

def gradf(A, C, x, b, lam):
    acT = np.transpose(A @ C)
    return 2 * (acT @ A @ C @ x) - 2 * (acT @ b) +lam

def Armijio(A, C, x, b, lam, gradx, d, a, beta, c, maxiter):
    for i in range(maxiter):
        obj_a = f(A,C,  np.clip(x + a*d, 0, None), b, lam)
        if obj_a < (f(A,C, np.clip(x, 0, None), b, lam) + c*a*np.inner(gradx,d)):
            return a
        else:
            a = beta*a
    return a


def gradient_descent(A, C, x, b, lam, maxIter, a0, beta, c, epsilon):
    f_val =[]
    x_k = []
    print(lam)
    for i in range(maxIter):
        grad = gradf(A,C, x, b, lam)
        d = -grad
        alpha = Armijio(A, C, x, b,lam,  grad, d, a0, beta, c, maxIter)
        xprev = x
        x = x + alpha * d
        x = np.clip(x, 0, None)
        x_k.append(C@x)
        f_val.append(f(A,C, x, b, lam))
        if (i > 0) and (LA.norm(x - xprev) / LA.norm(xprev) < epsilon):
            break
    return (C@x), f_val, x_k



x = sp.random(200,1, 0.1).toarray()
x = np.reshape(x,(200,))
A = np.random.normal(0,10,size = (100,200))


mu = np.random.normal(0, 0.1,100)

w0 = np.zeros(400)
I1 = np.eye(200)
I2 = -1 * np.eye(200)
C = np.concatenate((I1, I2), axis=1)
b = (A@x) + mu

alpha = 0.25
beta = 0.5
c = 1e-4
epsilon = 0.00001

xstar, f, x_k = gradient_descent(A, C, w0, b,50, 2000, alpha, beta, c, epsilon)
xnorm = [LA.norm(x-w) for w in x_k]
print(np.count_nonzero(xstar)/200)


plt.figure()
plt.plot(np.arange(len(f)),f)
plt.ylabel(r'$f(x_k)$')
plt.xlabel("iterations")
plt.title("f")
plt.show()

plt.figure()
plt.plot(np.arange(len(xnorm)),xnorm)
plt.xlabel("iterations")
plt.ylabel(r'$||x-x_k||_2$')
plt.title(r'$||x-x^*||_2$')
plt.show()

