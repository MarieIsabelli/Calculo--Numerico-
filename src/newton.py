"""
Método de Newton-Raphson
========================
Utiliza a reta tangente à curva no ponto atual para estimar a próxima
aproximação da raiz:

    x_{n+1} = x_n - f(x_n) / f'(x_n)

Converge quadraticamente (ordem 2) perto da raiz, quando f'(raiz) ≠ 0.
É o método mais rápido dos quatro, porém exige f' e pode divergir se
a chute inicial estiver longe da raiz ou se f' ≈ 0.
"""

import numpy as np
import pandas as pd


# ── Derivada numérica ─────────────────────────────────────────────────────────

def _derivada_numerica(f, x: float, h: float = 1e-7) -> float:
    """Diferença centrada de segunda ordem: O(h²)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


# ── Newton clássico ───────────────────────────────────────────────────────────

def newton(f, x0: float, df=None, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Newton-Raphson com derivada analítica ou numérica (automática).

    Parâmetros
    ----------
    f        : callable   — f(x)
    x0       : float      — chute inicial
    df       : callable | None  — f'(x); se None usa diferenças finitas
    tol      : float      — critério |x_{n+1} - x_n| < tol
    max_iter : int

    Retorna
    -------
    dict com raiz, iteracoes, convergiu, historico (DataFrame), erro_final.
    """
    usar_numerica = df is None
    _df = df if df is not None else lambda x: _derivada_numerica(f, x)

    x = x0
    historico = []
    convergiu = False
    erro = float("inf")

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = _df(x)

        if abs(dfx) < 1e-14:
            raise ZeroDivisionError(
                f"f'(x) ≈ 0 na iteração {i} (x={x:.6f}). "
                "Método diverge — tente outro x₀."
            )

        x_new = x - fx / dfx
        erro = abs(x_new - x)

        historico.append({
            "iteração": i,
            "x_n": x,
            "f(x_n)": fx,
            "f'(x_n)": dfx,
            "x_{n+1}": x_new,
            "erro |Δx|": erro,
        })

        x = x_new

        if erro < tol:
            convergiu = True
            break

    df_hist = pd.DataFrame(historico)

    return {
        "raiz": x,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": df_hist,
        "erro_final": erro,
        "metodo": "Newton-Raphson" + (" (df numérica)" if usar_numerica else ""),
        "derivada_analitica": not usar_numerica,
    }


# ── Newton para raízes múltiplas ──────────────────────────────────────────────

def newton_multiplas_raizes(f, df, d2f, x0: float, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Versão modificada para raízes de multiplicidade m > 1:

        x_{n+1} = x_n - f(x_n)*f'(x_n) / [f'(x_n)² - f(x_n)*f''(x_n)]

    Recupera convergência quadrática mesmo em raízes múltiplas.
    """
    x = x0
    historico = []
    convergiu = False
    erro = float("inf")

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        denom = dfx**2 - fx * d2fx

        if abs(denom) < 1e-14:
            raise ZeroDivisionError(f"Denominador nulo na iteração {i}.")

        x_new = x - fx * dfx / denom
        erro = abs(x_new - x)

        historico.append({"iteração": i, "x_n": x, "f(x_n)": fx, "erro": erro})
        x = x_new

        if erro < tol:
            convergiu = True
            break

    return {
        "raiz": x,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": pd.DataFrame(historico),
        "erro_final": erro,
        "metodo": "Newton (raízes múltiplas)",
    }


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # f(x) = x³ - x - 2  →  raiz em x ≈ 1.5213797
    def f(x):  return x**3 - x - 2
    def df(x): return 3*x**2 - 1

    print("=" * 60)
    print("  Newton-Raphson — derivada analítica")
    print("=" * 60)
    r = newton(f, x0=1.5, df=df, tol=1e-10)
    print(f"  Raiz : {r['raiz']:.12f}")
    print(f"  f(r) : {f(r['raiz']):.3e}")
    print(f"  Iter : {r['iteracoes']}")
    print(f"  Erro : {r['erro_final']:.3e}")
    print()
    print(r["historico"].to_string(index=False))

    print()
    print("=" * 60)
    print("  Newton-Raphson — derivada numérica (sem df)")
    print("=" * 60)
    r2 = newton(f, x0=1.5, tol=1e-10)
    print(f"  Raiz : {r2['raiz']:.12f}  |  Iter: {r2['iteracoes']}")
