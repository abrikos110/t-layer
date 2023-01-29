'''
R(h, t) = (t^n A(h/t) - A(h)) / (t^n - 1)

h = b
a = h/t
t = b/a
R(a, b) = ((b/a)^n A(a) - A(b)) / ((b/a)^n - 1)

p = 1/a
q = 1/b
R(n, m) = ((p/q)^n A(p) - A(q)) / ((p/q)^n - 1)
'''

# rounding errors can play a BIG role
# Richardson extrapolation
def int_RE(p, A, n=1): # p, p+1
    q = int((p+0) * (2*p)/(p)) * (p/p)
    t_n = (p/q) ** n
    return (t_n * A(p) - A(q)) / (t_n - 1)



if __name__ == '__main__':
    # test
    from fractions import Fraction as F
    I = F(1, 1)
    BIG = (I * 2) ** 600
    f = lambda x: BIG * I / x
    g = lambda n: (f(1 + I/n) - f(I)) *n

    g1 = lambda n: g(n)
    g2 = lambda n: int_RE(n, g1, 1)
    g3 = lambda n: int_RE(n, g2, 2)
    g4 = lambda n: int_RE(n, g3, 3)
    g5 = lambda n: int_RE(n, g4, 4)
    g6 = lambda n: int_RE(n, g5, 5)

    gg = lambda x: g6(x) / BIG
    order = 6

    err = lambda f, n: (1 + f(2+n))
    print(g4(3**10 * I).denominator)
    print(*[float(err(gg, I*3**i) * (3**i)**order) for i in range(10)], sep='\n  ')
    print()
    print(*[err(gg, 3**i) * (3**i)**order for i in range(10)], sep='\n  ')
