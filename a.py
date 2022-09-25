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
    d2 /= dx*dx
    d2[0] = d2[-1] = 0

    ans += dt * (d2 + f(x, t))  # FIRST ORDER TIME STEP
    #ans[0], ans[-1] = 1, 0
    ans[0] = ans[1]
    ans[-1] = ans[-2]

    return ans


def nsteps(x, y, t, dx, dt, f, n):
    ans = y.copy()
    for i in range(n):
        ans = step(x, ans, t + i*dt/n, dx, dt/n, f)
    return ans
def runge_step_time(x, y, t, dx, dt, f, n, maxn=1e4, eps=1e-3):
    y1 = nsteps(x, y, t, dx, dt, f, n)
    y2 = nsteps(x, y, t, dx, dt, f, n*2)
    while (y1 - y2).max() < eps / 2 and n > 1:
        y2 = y1
        n = n // 2
        y1 = nsteps(x, y, t, dx, dt, f, n)
    while (y1 - y2).max() > eps and 2*n <= maxn:
        y1 = y2
        n *= 2
        y2 = nsteps(x, y, t, dx, dt, f, n*2)

    return n, y2


zero = lambda *args: 0

n, nt = 10**2, 10**9
x = numpy.linspace(0, 1, n)
y = (x < 2/3) * (x > 1/3) * 1.0

dx = 1/n
dt = 0.5 * dx * dx

lst = y
t = 0
n = 1

plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))

line1, = ax.plot(x, y)
figure.canvas.flush_events()
figure.canvas.draw()
figure.canvas.flush_events()

time.sleep(1)
for i in range(nt):
    n, a = runge_step_time(x, lst, i*dt, dx, dt, zero, n, eps=1e-6)
    lst = a
    time.sleep(0.01)
    if time.time() - t > 0.03:
        print(dt/n, n, i*dt)
        line1.set_ydata(lst)
        figure.canvas.draw()
        figure.canvas.flush_events()
        t = time.time()
