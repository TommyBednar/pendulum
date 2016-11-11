from __future__ import division
import math
import matplotlib.pyplot as plt
import numpy as np


def _init(u0, t0, tf, h):
    n = int((tf-t0)/h)
    t = np.linspace(t0, tf, n+1)
    u = np.zeros((t.size, 2))
    u[0] = u0
    return n, t, u

# u[n+3] = u[n+2] + 1/12*h*(23*f[n+2] - 16*f[n+1] + 5*f[n])
def ab3(f, u0, t0, tf, h):
    n, t, u = _init(u0, t0, tf, h)
    u[1] = u[0] + h*f(u[0])
    u[2] = u[1] + h*f(u[1])

    for i in range(n-2):
        u[i+3] = u[i+2] + 1/12*h*(23*f(u[i+2]) - 16*f(u[i+1]) + 5*f(u[i]))
    return t, u

# v[n+2] = u[n] + 2*h*f(v[n+1])
# u[n+1] = v[n+1] + nu/2*(v[n+2] - 2*v[n+1] + u[n])
def lfra(f, u0, t0, tf, h):
    n, t, u = _init(u0, t0, tf, h)
    v = np.zeros((u.size+1, 2))
    v[1] = u[0] + h*f(u[0])
    nu = 0.8

    for i in range(n):
        v[i+2] = u[i] + 2*h*f(v[i+1])
        u[i+1] = v[i+1] + nu/2*(v[i+2] - 2*v[i+1] + u[i])

    return t, u

# w[n+2] = u[n] + 2*h*f(v[n+1])
# u[n+1] = v[n+1] + nu*alpha/2*(w[n+2] - 2*v[n+1] + u[n])
# v[n+2] = w[n+2] - nu*(1-alpha)/2*(w[n+2] - 2*v[n+1] + u[n])
def lfraw(f, u0, t0, tf, h):
    n, t, u = _init(u0, t0, tf, h)
    w = np.zeros((u.size+1, 2))
    v = np.zeros((u.size+1, 2))
    v[1] = u[0] + h*f(u[0])

    nu = 0.8
    alpha = 0.53

    for i in range(n):
        w[i+2] = u[i] + 2*h*f(v[i+1])
        u[i+1] = v[i+1] + nu*alpha/2*(w[i+2] - 2*v[i+1] + u[i])
        v[i+2] = w[i+2] - nu*(1-alpha)/2*(w[i+2] - 2*v[i+1] + u[i])

    return t, u

# v[n+3] = u[n+1] + 2*h*f(v[n+2])
# u[n+2] = v[n+2] + beta/2*(v[n+3] - 2*v[n+2] + u[n+1]) - beta/2*(v[n+2] - 2*u[n+1] + u[n])
def lfhora(f, u0, t0, tf, h):
    n, t, u = _init(u0, t0, tf, h)
    v = np.zeros((u.size+1, 2))
    u[1] = u[0] + h*f(u[0])
    v[2] = u[1] + h*f(u[1])

    beta = 0.4

    for i in range(n-1):
        v[i+3] = u[i+1] + 2*h*f(v[i+2])
        u[i+2] = v[i+2] + beta/2*(v[i+3] - 2*v[i+2] + u[i+1]) - beta/2*(v[i+2] - 2*u[i+1] + u[i])

    return t, u


# f(u) = du/dt where u = [theta, v]
# dtheta/dt = v/L
# dv/dt = -g*sin(theta)
def f(u):
    return np.array([u[1]/49, -9.8*math.sin(u[0])])


def main():
    u0 = np.array([0.9*math.pi, 0])
    t0 = 0
    tf = 200
    h = 0.1

    for method in [ab3, lfra, lfraw, lfhora]:
        t, u = method(f, u0, t0, tf, h)

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel(u"\u03B8")
        plt.plot(t, u[:,0], label=method.__name__)
        plt.legend()


        plt.figure(2)
        plt.xlabel("t")
        plt.ylabel("v")
        plt.plot(t, u[:,1], label=method.__name__)
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
