import numpy
import matplotlib.pyplot as plt

'''
y'_t = y''_xx + f(x, t)
y'_x (0, t) = y'_x (1, t) = 0
y(x, 0) = ...
'''

def step(x, y, t, dx, dt, f):
    R = numpy.roll

    ans = y.copy()
    d2 = R(y, -1) - 2*y + R(y, 1)
    d2[0] = d2[-1] = 0

    ans += dt * d2 + f(x, t)  # FIRST ORDER TIME STEP
    ans[0], ans[-1] = ans[1], ans[-2]  # FIRST ORDER BOUNDARY

    return ans, d2


zero = lambda *args: 0

n, nt = 10**2, 10**9
x = numpy.linspace(0, 1, n)
y = (x < 0.5) * 1.0

t = 1000
dx, dt = 1/n, t/nt

lst = y
for i in range(nt):
    a, d = step(x, lst, i*dt, dx, dt, zero)
    lst = a
#print(ans[::10])
    if i % 100000 == 0:
        print(dt)
        plt.plot(x, lst)
        plt.show()
    dt = 0.001 / abs(d).max()
    dt = min(0.5 * dx*dx, dt)
