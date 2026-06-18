"""
Método da Secante
=================
Aproxima f'(x) por diferença finita usando os dois pontos anteriores,
eliminando a necessidade de calcular a derivada analítica:

    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

Ordem de convergência: φ ≈ 1.618 (razão áurea) — superlinear.
Requer dois pontos iniciais (x0, x1) em vez de um intervalo.
"""

import numpy as np
import pandas as pd


def secante(f, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Método da Secante para encontrar raíz de f.

    Parâmetros
    ----------
    f        : callable
    x0, x1  : float     — dois pontos iniciais (não precisam conter raiz)
    tol      : float     — critério |x_{n+1} - x_n| < tol
    max_iter : int

    Retorna
    -------
    dict com raiz, iteracoes, convergiu, historico (DataFrame), erro_final.
    """
    historico = []
    convergiu = False
    erro = float("inf")

    x_prev, x_curr = x0, x1
    f_prev, f_curr = f(x0), f(x1)

    for i in range(1, max_iter + 1):
        denom = f_curr - f_prev

        if abs(denom) < 1e-14:
            raise ZeroDivisionError(
                f"f(x_n) - f(x_{{n-1}}) ≈ 0 na iteração {i}. "
                "Escolha pontos iniciais mais afastados."
            )

        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        f_new = f(x_new)
        erro = abs(x_new - x_curr)

        historico.append({
            "iteração": i,
            "x_{n-1}": x_prev,
            "x_n": x_curr,
            "x_{n+1}": x_new,
            "f(x_{n+1})": f_new,
            "erro |Δx|": erro,
        })

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new

        if erro < tol:
            convergiu = True
            break

    df_hist = pd.DataFrame(historico)

    return {
        "raiz": x_curr,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": df_hist,
        "erro_final": erro,
        "metodo": "Secante",
    }


def secante_brent(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Método de Brent — combina bisseção, interpolação quadrática inversa e
    secante para garantir convergência global com velocidade superlinear.
    É o padrão em bibliotecas científicas (scipy.optimize.brentq).
    """
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos para Brent.")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    s = b
    d = 0.0
    historico = []
    convergiu = False
    erro = float("inf")

    for i in range(1, max_iter + 1):
        if fa != fc and fb != fc:
            # Interpolação quadrática inversa
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secante
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2

        if cond1 or cond2 or cond3:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        erro = abs(b - a)
        historico.append({"iteração": i, "a": a, "b": b, "s": s, "f(s)": fs, "erro": erro})

        d, c, fc = c, b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(fb) < tol or erro < tol:
            convergiu = True
            break

    return {
        "raiz": b,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": pd.DataFrame(historico),
        "erro_final": erro,
        "metodo": "Brent",
    }


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def f(x): return x**3 - x - 2

    print("=" * 60)
    print("  Método da Secante")
    print("=" * 60)
    r = secante(f, x0=1.0, x1=2.0, tol=1e-10)
    print(f"  Raiz : {r['raiz']:.12f}")
    print(f"  f(r) : {f(r['raiz']):.3e}")
    print(f"  Iter : {r['iteracoes']}")
    print(f"  Erro : {r['erro_final']:.3e}")
    print()
    print(r["historico"].to_string(index=False))

    print()
    print("=" * 60)
    print("  Método de Brent (referência SciPy)")
    print("=" * 60)
    rb = secante_brent(f, 1.0, 2.0, tol=1e-10)
    print(f"  Raiz : {rb['raiz']:.12f}  |  Iter: {rb['iteracoes']}")
