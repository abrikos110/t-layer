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
    y = y.copy()
    Q = q
    K = k
    if callable(q): Q = q(y)
    if callable(k): K = k(y)

    # Y[i] = y[i] in time t-dt
    # ( (K[i+1]+K[i])/2 * (y[i+1] - y[i]) - (K[i]+K[i-1])/2 * (y[i] - y[i-1]) ) / (dx*dx) + Q = (y[i] - Y[i]) / dt

    Km1 = numpy.roll(K, 1)
    Kp1 = numpy.roll(K, -1)
    n = y.shape[0]

    A = (K + Km1) / 2 / (dx*dx)
    B = -(Kp1 + 2*K + Km1) / 2 / (dx*dx) - 1/dt
    C = (Kp1 + K) / 2 / (dx*dx)
    F = -Q - y / dt

    A[0] = C[0] = F[0] = 0
    B[0] = 1
    A[n-1] = C[n-1] = F[n-1] = 0
    B[n-1] = 1
    # A[i] y[i-1] + B[i] y[i] + C[i] x[i+1] = F[i]
    alpha = numpy.ones(n+1) * 1e100
    beta = numpy.ones(n+1) * 1e100
    alpha[1] = -C[0] / B[0]
    beta[1] = F[0] / B[0]
    for i in range(1, n):
        den = A[i] * alpha[i] + B[i]
        alpha[i+1] = -C[i] / den
        beta[i+1] = (F[i] - A[i] * beta[i]) / den
    y[n-1] = (F[-1] - A[-1]*beta[-1]) / (B[-1] + A[-1]*alpha[-1])
    for i in range(n - 2, 0, -1):
        y[i] = alpha[i+1] * y[i+1] + beta[i+1]

    return y

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
    noise = numpy.random.rand(*y.shape) * eps / 2
    y1 = nsteps(x, y + noise, t, dx, dt, k, q, 1, sf)
    y2 = nsteps(x, y, t, dx, dt, k, q, 2, sf)

    while (y1 - y2).max() < eps/2 and dt < 1:
        dt *= 2
        y1 = nsteps(x, y + noise, t, dx, dt, k, q, 1, sf)
        y2 = nsteps(x, y, t, dx, dt, k, q, 2, sf)

    while ((y1 - y2).max() >= eps or not numpy.isfinite(y2).all()) and dt > 0:
        dt /= 2
        y1 = nsteps(x, y + 2*noise, t, dx, dt, k, q, 1, sf)
        y2 = nsteps(x, y, t, dx, dt, k, q, 2, sf)

    return dt/2, y2


def main(tp='exp', n=10**2, nt=10**9, sleep_time=0.01):
    if tp == 'exp':
        sf = exp_step
    elif tp == 'lin':
        sf = step
    elif tp == 'imp':
        sf = implicit_step
    else:
        raise Exception

    k = lambda y: y*0 + 1
    q = lambda y: 0*y

    k = lambda y:     y**2.0
    q = lambda y: 30* y**3.0

    x = numpy.linspace(0, 1, n)
    y = (x < 0.6) * (x > 0.4) * 15.

    dx = 1/n
    dt = 1e-4 * dx*dx#1e-4 * dx*dx

    lst = y
    t = 0

    dt_hist = [dt]

    plt.ion()
    figure, ax = plt.subplots(figsize=(16,9))
    line1, = ax.plot(x, y, label=tp)
    line2, = ax.plot(dt_hist, label='dt')

    plt.legend()
    figure.canvas.flush_events()
    figure.canvas.draw()
    figure.canvas.flush_events()

    peak_dt = dt
    for i in range(nt):
        if not plt.fignum_exists(figure.number):
            break
        dt, a = runge_rule(x, lst, t, dx, dt, k, q, eps=dt * 1e6, sf=sf)
        lst = a
        t += dt
        dt_hist.append(dt)

        peak_dt = max(peak_dt, dt)
        plt.title('t={} peak_dt={} dt={}'.format(round(t, 13), round(peak_dt, 10), dt))
        time.sleep(sleep_time)

        line1.set_ydata(lst)
        ax.set_ylim(lst.min(), lst.max() * 1.1)

        dh = numpy.array(dt_hist)
        line2.set_xdata(numpy.linspace(0, 1, dh.shape[0]))
        line2.set_ydata(dh / dh.max() * lst.max() * 0.9)

        figure.canvas.draw()
        figure.canvas.flush_events()


if __name__ == '__main__':
    from sys import argv
    tp = 'exp'
    sleep_time = 0.01
    if len(argv) > 1:
        tp = argv[1]
        if len(argv) > 2:
            sleep_time = float(argv[2])
    main(tp, sleep_time=sleep_time)

raise SystemExit
