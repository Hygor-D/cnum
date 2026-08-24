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