"""Algoritmos numéricos usados nas atividades."""

import numpy as np

try:
    from scipy.optimize import approx_fprime
except ImportError:  # pragma: no cover - fallback para ambientes sem scipy
    approx_fprime = None


def JN(x, F, eps=1e-8):
    """Aproxima a matriz jacobiana de F em x por diferenças finitas."""
    x = np.asarray(x, dtype=float)
    n = x.size
    Jnum = np.zeros((n, n), dtype=float)

    if approx_fprime is not None:
        for i in range(n):
            def fi(v, idx=i):
                return F(v)[idx]
            Jnum[i, :] = approx_fprime(x, fi, epsilon=eps)
        return Jnum

    for i in range(n):
        for j in range(n):
            h = eps * max(1.0, abs(x[j]))
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            Jnum[i, j] = (F(xp)[i] - F(xm)[i]) / (2.0 * h)

    return Jnum


def G(x, F, J):
    """Iteração de Newton em forma de ponto fixo para sistemas não lineares."""
    x = np.asarray(x, dtype=float)
    return x - np.linalg.inv(J(x)) @ F(x)


def GN(x, F):
    """Iteração de Newton usando jacobiano numérico aproximado."""
    x = np.asarray(x, dtype=float)
    return x - np.linalg.inv(JN(x, F)) @ F(x)


def fixed_point(a, g, TOL=1e-8, iter=1000):
    """Encontra um ponto fixo da função g a partir de a."""
    x = g(a)
    i = 1
    while np.linalg.norm(x - a) > TOL and i < iter:
        a = x
        x = g(a)
        i += 1
    return x


def newton_raphson_sistema(x0, F, J=None, TOL=1e-8, iter=100, eps=1e-8):
    """Resolve F(x)=0 usando Newton-Raphson para sistemas não lineares."""
    x = np.asarray(x0, dtype=float).copy()
    if J is None:
        J = lambda v: JN(v, F, eps=eps)

    for i in range(1, iter + 1):
        Fx = np.asarray(F(x), dtype=float)
        if np.linalg.norm(Fx, ord=np.inf) <= TOL:
            return x, i

        Jx = np.asarray(J(x), dtype=float)
        try:
            delta = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(Jx) @ Fx

        x_new = x - delta
        if np.linalg.norm(x_new - x, ord=np.inf) <= TOL:
            return x_new, i
        x = x_new

    return x, iter


def bissecao(f, a, b, TOL, iter=16):
    """Encontra uma raiz de f em [a, b] pelo método da bisseção."""
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")

    for i in range(1, iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        if fc == 0 or abs(b - a) / 2 <= TOL:
            return c, i

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return c, iter


def pontofixo(a, g, TOL=1e-8, iter=100):
    """Encontra um ponto fixo de g a partir de a."""
    if TOL <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    if iter <= 0:
        raise ValueError("O número de iterações deve ser positivo.")

    for i in range(1, iter + 1):
        x = g(a)
        if abs(x - a) <= TOL:
            return x, i
        a = x

    return a, iter


def newton_raphson(a, f, TOL=1e-8, df=None, iter=100):
    """Encontra uma raiz pelo método de Newton-Raphson."""
    if df is None:
        def df(x):
            h = 1e-8 * max(1.0, abs(x))
            return (f(x + h) - f(x - h)) / (2.0 * h)

    def g(x):
        derivative = df(x)
        if derivative == 0:
            raise ZeroDivisionError("A derivada se anulou durante a iteração.")
        return x - f(x) / derivative

    return pontofixo(a, g, TOL, iter)


def secante(a, b, f, TOL=1e-8, iter=100):
    """Encontra uma raiz pelo método da secante."""
    if TOL <= 0:
        raise ValueError("A tolerância deve ser positiva.")
    if iter <= 0:
        raise ValueError("O número de iterações deve ser positivo.")

    for i in range(1, iter + 1):
        fa = f(a)
        fb = f(b)
        denominator = fb - fa
        if denominator == 0:
            raise ZeroDivisionError("As avaliações da função são iguais.")
        x = (a * fb - b * fa) / denominator
        if abs(x - b) <= TOL:
            return x, i
        a, b = b, x

    return b, iter