"""Algoritmos numéricos usados nas atividades."""


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