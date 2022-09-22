import numpy
import matplotlib.pyplot as plt
import time

'''
y'_t = y''_xx + f(x, t)
y'_x (0, t) = 1
y'_x (1, t) = 0
y(x, 0) = [x < 0.5]
'''

def step(x, y, t, dx, dt, f):
    R = numpy.roll

    ans = y.copy()
    d2 = R(y, -1) - 2*y + R(y, 1)
    d2[0] = d2[-1] = 0

    ans += dt * d2 + f(x, t)  # FIRST ORDER TIME STEP
    #ans[0], ans[-1] = 1, 0
    ans[0] = ans[1]
    ans[-1] = ans[-2]

    return ans, d2


zero = lambda *args: 0

n, nt = 10**2, 10**9
x = numpy.linspace(0, 1, n)
y = (x < 0.5) * 1.0

dx = 1/n
dt = 0.5 * dx * dx

lst = y
t = 0

plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)
figure.canvas.draw()
figure.canvas.flush_events()
for i in range(nt):
    a, d = step(x, lst, i*dt, dx, dt, zero)
    lst = a
    if time.time() - t > 0.1:
        print(dt, i, i*dt)
        line1.set_ydata(lst)
        figure.canvas.draw()
        figure.canvas.flush_events()
        t = time.time()
