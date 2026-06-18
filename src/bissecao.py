"""
Método da Bisseção
==================
Divide repetidamente o intervalo [a, b] ao meio, mantendo o subintervalo
onde a função troca de sinal. Converge linearmente (ordem 1).

Requisito: f(a) * f(b) < 0  (Teorema de Bolzano)
"""

import numpy as np
import pandas as pd


def bissecao(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Encontra uma raiz de f no intervalo [a, b] pelo método da bisseção.

    Parâmetros
    ----------
    f        : callable  — função contínua
    a, b     : float     — extremos do intervalo
    tol      : float     — tolerância para critério de parada (|b-a|/2)
    max_iter : int       — número máximo de iterações

    Retorna
    -------
    dict com chaves:
        raiz       : float          — aproximação da raiz
        iteracoes  : int            — número de iterações realizadas
        convergiu  : bool           — True se o critério foi satisfeito
        historico  : pd.DataFrame   — tabela com a, b, c, f(c), erro por iteração
        erro_final : float          — erro na última iteração
    """
    fa = f(a)

    if fa * f(b) > 0:
        raise ValueError(
            f"f(a)={fa:.4f} e f(b)={f(b):.4f} têm o mesmo sinal. "
            "Escolha um intervalo que contenha a raiz."
        )

    historico = []
    convergiu = False

    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        erro = abs(b - a) / 2.0

        historico.append({
            "iteração": i,
            "a": a,
            "b": b,
            "c (raiz aprox.)": c,
            "f(c)": fc,
            "erro |b-a|/2": erro,
        })

        if erro < tol or abs(fc) < 1e-14:
            convergiu = True
            break

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

    df = pd.DataFrame(historico)

    return {
        "raiz": c,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": df,
        "erro_final": erro,
        "metodo": "Bisseção",
    }


def bissecao_teorico_iter(a: float, b: float, tol: float) -> int:
    """Número mínimo teórico de iterações: ceil(log2((b-a)/tol))."""
    return int(np.ceil(np.log2(abs(b - a) / tol)))


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def f(x):
        return x**3 - x - 2

    resultado = bissecao(f, a=1.0, b=2.0, tol=1e-8)

    print("=" * 55)
    print(f"  Método da Bisseção")
    print("=" * 55)
    print(f"  Raiz encontrada : {resultado['raiz']:.10f}")
    print(f"  f(raiz)         : {f(resultado['raiz']):.2e}")
    print(f"  Iterações       : {resultado['iteracoes']}")
    print(f"  Erro final      : {resultado['erro_final']:.2e}")
    print(f"  Convergiu       : {resultado['convergiu']}")
    print(f"  Iter. teórico   : {bissecao_teorico_iter(1, 2, 1e-8)}")
    print()
    print(resultado["historico"].to_string(index=False))
