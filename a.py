import numpy
import matplotlib.pyplot as plt
import time


def exponent(linear_operator, t, u, eps=1e-9):
    ''' exp(A, t, u) = exp(A t) @ u '''
    x = u
    ans = 0
    i = 0
    f = 1
    for i in range(32):#while abs((ans + x) - ans).max() > eps:
        ans += x
        i += 1
        xx = linear_operator(x) * t / i
        x = xx
    return ans


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
#ans[0], ans[-1] = y[0], y[-1]
    ans[0], ans[-1] = ans[1], ans[-2]

    return ans

def exp_step(x, y, t, dx, dt):
    ee = exponent(lambda y: step(x, y, t, dx, 1, lambda a, b: 0*a) - y, dt, y)
    return ee


def nsteps(x, y, t, dx, dt, f, n, tp=1):
    ans = y.copy()
    if tp:
        for i in range(n):
            ans = exp_step(x, ans, t + i*dt/n, dx, dt/n)
    else:
        for i in range(n):
            ans = step(x, ans, t + i*dt/n, dx, dt/n, f)
    return ans
def runge_step_time(x, y, t, dx, dt, f, n, maxn=1e+10, eps=1e-3, tp=1):
    y1 = nsteps(x, y, t, dx, dt, f, n, tp)
    y2 = nsteps(x, y, t, dx, dt, f, n*2, tp)
    while (y1 - y2).max() < eps / 2 and n > 1:
        y2 = y1
        n = n // 2
        y1 = nsteps(x, y, t, dx, dt, f, n, tp)
    while ((y1 - y2).max() > eps or not numpy.isfinite(y1-y2).all()) and 2*n <= maxn:
        y1 = y2
        n *= 2
        y2 = nsteps(x, y, t, dx, dt, f, n*2, tp)

    return n, y2


zero = lambda *args: 0

n, nt = 10**2, 10**9
x = numpy.linspace(0, 1, n)
y = ((x * 2).round() % 2) * 1.0

dx = 1/n
dt = dx
#dt = 0.5 * dx * dx

lsta = lstb = y
t = 0
na = nb = 1

plt.ion()
figure, ax = plt.subplots(figsize=(16, 9))

line1, = ax.plot(x, y, label='exp')
line2, = ax.plot(x, y, label='usual')
plt.legend()
figure.canvas.flush_events()
figure.canvas.draw()
figure.canvas.flush_events()

for i in range(nt):
    if not plt.fignum_exists(figure.number):
        break
    na, a = runge_step_time(x, lsta, i*dt, dx, dt, zero, na, eps=1e-5, maxn=10000, tp=1)
    nb, b = runge_step_time(x, lstb, i*dt, dx, dt, zero, nb, eps=1e-5, maxn=10000, tp=0)
    lsta = a
    lstb = b
    time.sleep(0.03)
    if time.time() - t > 0.03:
        print(dt/na, na, ';', dt/nb, nb, i*dt, abs(lsta).max())
        line1.set_ydata(lsta)
        line2.set_ydata(lstb)
        figure.canvas.draw()
        figure.canvas.flush_events()
        t = time.time()
