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


def implicit_step(x, y, t, dx, dt, k, q):
    Q = q
    K = k
    if callable(q): Q = q(y)
    if callable(k): K = k(y)

    # TODO: metod progonki

    # Y[i] = y[i] in time t-dt
    # ( (K[i+1]+K[i])/2 * (y[i+1] - y[i]) - (K[i]+K[i-1])/2 * (y[i] - y[i-1]) ) / (dx*dx) + Q = (y[i] - Y[i]) / dt
    # y[i+1] * (K[i+1]+K[i]) / (2*dx*dx) + y[i] * (-(K[i+1]+2K[i]+K[i-1]) / (2*dx*dx) - 1/dt) ...

def step(x, y, t, dx, dt, k, q):
    ''' u'_t = div(k(u) grad u) + q(u) = (k(u) u'_x)'_x'''
    Q = q
    K = k
    if callable(q): Q = q(y)
    if callable(k): K = k(y)

    R = numpy.roll

    ans = y.copy()
    d2 = (R(K, -1) + K) / 2 * (R(y, -1) - y) - (K + R(K, 1)) / 2 * (y - R(y, 1))
    d2 /= dx*dx
    d2[0] = d2[-1] = 0

    ans += dt * (d2 + Q)
#ans[0], ans[-1] = y[0], y[-1]
#ans[0], ans[-1] = ans[1], ans[-2] # first order
    ans[0] = ans[-1] = 0

    return ans

def exp_step(x, y, t, dx, dt, k, q):
    K = k(y)
    Q = q(y)
    ee = exponent(lambda y: step(x, y, t, dx, 1, K, Q) - y, dt, y)
    return ee

def nsteps(x, y, t, dx, dt, k, q, n, sf):
    ans = y.copy()
    for i in range(n):
        ans = sf(x, ans, t+i*dt/n, dx, dt/n, k, q)
    return ans

def runge_rule(x, y, t, dx, dt, k, q, eps, sf):
    y1 = nsteps(x, y, t, dx, dt, k, q, 1, sf)
    y2 = nsteps(x, y, t, dx, dt, k, q, 2, sf)

    while (y1 - y2).max() < eps/2:
        dt *= 2
        y1 = nsteps(x, y, t, dx, dt, k, q, 1, sf)
        y2 = nsteps(x, y, t, dx, dt, k, q, 2, sf)

    while (y1 - y2).max() >= eps or not numpy.isfinite(y2).all():
        dt /= 2
        y1 = nsteps(x, y, t, dx, dt, k, q, 1, sf)
        y2 = nsteps(x, y, t, dx, dt, k, q, 2, sf)

    return dt/2, y2


def main(tp='exp', n=10**2, nt=10**9):
    if tp == 'exp':
        sf = exp_step
    elif tp == 'lin':
        sf = step
    elif tp == 'imp':
        sf = implicit_step
    else:
        raise Exception

    k = lambda y:      y**2.0 #u*0 + 1
    q = lambda y: 30 * y**3.0 #0*u
    x = numpy.linspace(0, 1, n)
    y = (x < 0.6) * (x > 0.4) * 15.

    dx = 1/n
    dt = 1e-4 * dx*dx

    lst = y
    t = 0

    dt_hist = [0]

    plt.ion()
    figure, ax = plt.subplots(figsize=(16,9))
    line1, = ax.plot(x, y, label=tp)
    line2, = ax.plot(dt_hist, label='dt')

    plt.legend()
    figure.canvas.flush_events()
    figure.canvas.draw()
    figure.canvas.flush_events()

    for i in range(nt):
        if not plt.fignum_exists(figure.number):
            break
        dt, a = runge_rule(x, lst, t, dx, dt, k, q, eps=dt * 1e6, sf=sf)
        lst = a
        t += dt
        dt_hist.append(dt)

        plt.title('t={} dt={}'.format(round(t, 13), dt))
        time.sleep(0.3)

        line1.set_ydata(lst)
        ax.set_ylim(lst.min(), lst.max() * 1.1)

        dh = numpy.array(dt_hist)
        line2.set_xdata(numpy.linspace(0, 1, dh.shape[0]))
        line2.set_ydata(dh / dh.max() * lst.max() * 0.9)

        figure.canvas.draw()
        figure.canvas.flush_events()


if __name__ == '__main__':
    main('exp')

raise SystemExit
