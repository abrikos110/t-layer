import numpy
import numpy as np
from numpy import sqrt, cos, pi, exp, sin
import matplotlib.pyplot as plt
import time


def exponent(linear_operator, t, u, eps=1e-9):
    ''' exp(A, t, u) = exp(A t) @ u '''
    x = u
    ans = 0
    i = 0
    f = 1
    for i in range(22):#while abs((ans + x) - ans).max() > eps:
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

def iter_imp_step(x, y, t, dx, dt, k, q):
    ans = y.copy()

    for i in range(9):
        ans = implicit_step(x, y, t, dx, dt, k(ans), q(ans))

    ans = implicit_step(x, y, t, dx, dt, (k(ans) + k(y))/2, (q(ans)+q(y))/2)

    return ans

def step(x, y, t, dx, dt, k, q):
    ''' u'_t = div(k(u) grad u) + q(u) = (k(u) u'_x)'_x'''
    from numpy import roll as R

    Q = q(y) if callable(q) else q
    K = k(y) if callable(k) else k

    ans = y.copy()
    d2 = (R(K, -1) + K) / 2 * (R(y, -1) - y) - (K + R(K, 1)) / 2 * (y - R(y, 1))
    d2 /= dx*dx
    d2[0] = d2[-1] = 0

    ans += dt * (d2 + Q)
#ans[0], ans[-1] = y[0], y[-1]
#ans[0], ans[-1] = ans[1], ans[-2] # first order
    ans[0] = ans[-1] = 0

    return ans

def superstep(x, y, t, dx, dt, k, q, k_diff, q_diff):
    from numpy import roll as R

    Q = q(y) if callable(q) else q
    K = k(y) if callable(k) else k
    QD = q_diff(y) if callable(q_diff) else q_diff
    KD = k_diff(y) if callable(k_diff) else k_diff
    f = lambda c, u: (R(c,-1)+c)/2 * (R(u,-1)-u) - (R(c,1)+c)/2 * (u-R(u,1))

    ans = y.copy()
    d1 = f(K, y) / (dx*dx) + Q
    d1[0] = d1[-1] = 0

    d2 = QD * d1 + (f(KD * d1, y) + f(K, d1)) / (dx*dx)

    ans += dt * d1 + dt**2/2 * d2
    ans[0] = ans[-1] = 0

    return ans

def estep(x, t, dx, dt, ykqdd, k, q):
    ans = step(x, ykqdd[0], t, dx, dt, ykqdd[1], ykqdd[2])
    return ans, k(ans), q(ans)
def exp_step(x, y, t, dx, dt, k, q):
    K = k(y)
    Q = q(y)
    ee = exponent(lambda ykq: estep(x, t, dx, dt, ykq, k, q) - ykq, 1, np.array([y, K, Q]))
    return ee[0]

def nsteps(x, y, t, dx, dt, k, q, n, sf, k_diff=None, q_diff=None):
    ans = y.copy()
    for i in range(n):
        if sf == superstep:
            ans = sf(x, ans, t+i*dt/n, dx, dt/n, k, q, k_diff, q_diff)
        else:
            ans = sf(x, ans, t+i*dt/n, dx, dt/n, k, q)
    return ans

def runge_rule(x, y, t, dx, dt, k, q, eps, sf, k_diff=None, q_diff=None):
    noises = [abs(numpy.random.rand(*y.shape)) * eps * 0.1 for i in range(2)]
    y1 = nsteps(x, y + noises[0], t, dx, dt, k, q, 1, sf, k_diff=k_diff, q_diff=q_diff)
    y2 = nsteps(x, y + noises[1], t, dx, dt, k, q, 2, sf, k_diff=k_diff, q_diff=q_diff)

    while 0 and abs(y1 - y2).max() < eps/2 and dt < 0.001:
        dt *= 2
        y1 = nsteps(x, y + noises[0], t, dx, dt, k, q, 1, sf, k_diff=k_diff, q_diff=q_diff)
        y2 = nsteps(x, y + noises[1], t, dx, dt, k, q, 2, sf, k_diff=k_diff, q_diff=q_diff)

    while (abs(y1 - y2).max() >= eps or not numpy.isfinite(y2).all()) and dt > 0:
        dt /= 2
        y1 = nsteps(x, y + noises[0], t, dx, dt, k, q, 1, sf, k_diff=k_diff, q_diff=q_diff)
        y2 = nsteps(x, y + noises[1], t, dx, dt, k, q, 2, sf, k_diff=k_diff, q_diff=q_diff)

    y2 = nsteps(x, y, t, dx, dt, k, q, 1, sf, k_diff=k_diff, q_diff=q_diff)
    return dt, y2


def main(tp='exp', n=10**2, nt=10**9, sleep_time=0.01):
    if tp == 'exp':
        sf = exp_step
    elif tp == 'lin':
        sf = step
    elif tp == 'imp':
        sf = implicit_step
    elif tp == 'superstep':
        sf = superstep
    elif tp == 'iter':
        sf = iter_imp_step
    else:
        raise Exception

    k = lambda y: y*0 + 1
    q = lambda y: 0*y
    k_diff = q_diff = lambda y: 0*y
    exact = lambda x, t: sin(x) * exp(-t)

    if 1:
        sigma = 1.0
        beta = sigma + 1

        q0 = 10
        k0 = (sigma ** 2 * q0 / (sigma + 1)) / np.pi ** 2
        q = lambda y: q0 * y**beta
        q_diff = lambda y: q0 * beta * y**(beta-1)
        k = lambda y: k0 * y**sigma
        k_diff = lambda y: k0 * sigma * y**(sigma-1)

        L_T = 2 * pi * sqrt(k0 / q0) * sqrt((sigma + 1) / (sigma*sigma))
        def exact(x, t, tf=0.05):
            x = x-pi/2
            coef = 2 * (sigma + 1) / (sigma*(sigma + 2))
            cosin = cos((pi * x) / L_T) ** 2
            ans = (q0 * (tf - t)) ** (-1./sigma) * (coef * cosin) ** (1./sigma)
            ans[abs(x) > L_T/2] = 0
            return ans

    x = numpy.linspace(0, pi, n)
    y = exact(x, 0)
    y0 = y.copy()

    dx = x[1] - x[0]
    dt = 0.4 * dx * dx

    lst = y
    t = 0

    dt_hist = [dt]

    plt.ion()
    figure, ax = plt.subplots(figsize=(16,9))
    lines = [ax.plot(x, y, label=tp)[0],
             ax.plot(dt_hist, label='dt')[0],
             ax.plot(x, exact(x, t), label="exact", linewidth=5.0, alpha=0.5)[0],
             ax.plot(x, y0, label='y0')[0]]
    data = [(lambda: x, lambda: lst),
            (lambda: np.linspace(0,1,len(dt_hist)),
             lambda: np.array(dt_hist) / max(dt_hist) * lst.max() * 0.9),
            (lambda: x, lambda: exact(x, t)),
            (lambda: x, lambda: y0)]

    plt.legend()
    figure.canvas.flush_events()
    figure.canvas.draw()
    figure.canvas.flush_events()
    time.sleep(1)

    peak_dt = dt
    for i in range(nt):
        print(t, abs(lst[len(y)//2] - exact(x[len(y)//2:][:1], t)[0]))
        if not plt.fignum_exists(figure.number):
            break
        dt, a = runge_rule(x, lst, t, dx, dt, k, q, k_diff=k_diff, q_diff=q_diff, eps=1e-3, sf=sf)
        lst = a
        t += dt
        dt_hist.append(dt)

        peak_dt = max(peak_dt, dt)
        plt.title('t={} peak_dt={} dt={}'.format(round(t, 13), round(peak_dt, 10), dt))
#time.sleep(sleep_time)

        ax.set_ylim(lst.min(), max(1, lst.max()) * 1.05)

        for i in range(len(lines)):
            lines[i].set_xdata(data[i][0]())
            lines[i].set_ydata(data[i][1]())

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
