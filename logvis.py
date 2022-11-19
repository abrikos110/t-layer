import numpy as np
import matplotlib.pyplot as plt

files = ['imp.log', 'lin.log', 'exp.log', 'exp2.log', 'exp3.log']
data = []

for f in files:
    with open(f) as f:
        *f, = f
        if not f[-1]:
            f.pop()
        *f, = map(lambda x: list(map(float, x.split())), f)
        for g in f:
            assert len(g) == 2, str(g)
        f = np.array(f).T
        f = f.T[f[0] < 0.05].T
        data.append(f)


def f(x):
    l, r = 0, data[2].shape[1]
    while r-l>1:
        q = float(data[2][0][(r+l)//2])
        if q <= x:
            l = (l+r)//2
        else:
            r = (r+l)//2
    return data[2][1][l]
for i in range(len(files)):
#plt.plot(data[i][0], np.log(data[i][1]) / np.log(np.array(list(map(f, data[i][0])))), label=files[i], linewidth=3-i)
    plt.yscale('log')
    plt.plot(data[i][0], data[i][1], label=files[i])

plt.legend()
plt.show()

for i in range(len(files)):
    plt.plot(data[i][0][:-1], (np.roll(data[i][0], -1) - data[i][0])[:-1], label=files[i])
plt.legend()
plt.show()
