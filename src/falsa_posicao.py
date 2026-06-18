"""
Método da Falsa Posição (Regula Falsi)
=======================================
Variante da bisseção que usa interpolação linear para calcular o ponto c,
tendendo a convergir mais rápido que a bisseção pura.

c = b - f(b) * (b - a) / (f(b) - f(a))

Converge superlinearmente em casos favoráveis, porém pode estagnar
(um extremo fica fixo). A versão Illinois (modificada) corrige isso.
"""

import numpy as np
import pandas as pd


def falsa_posicao(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Método da Falsa Posição (Regula Falsi) clássico.

    Parâmetros
    ----------
    f        : callable
    a, b     : float  — intervalo inicial com f(a)*f(b) < 0
    tol      : float  — tolerância |c_novo - c_anterior|
    max_iter : int

    Retorna
    -------
    dict com raiz, iteracoes, convergiu, historico (DataFrame), erro_final.
    """
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError(
            f"f(a)={fa:.4f} e f(b)={fb:.4f} têm o mesmo sinal. "
            "Forneça intervalo com troca de sinal."
        )

    historico = []
    c_prev = a
    convergiu = False

    for i in range(1, max_iter + 1):
        # ponto de intersecção da secante com o eixo x
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        erro = abs(c - c_prev) if i > 1 else abs(b - a)

        historico.append({
            "iteração": i,
            "a": a,
            "b": b,
            "c (raiz aprox.)": c,
            "f(c)": fc,
            "erro |Δc|": erro,
        })

        if i > 1 and erro < tol:
            convergiu = True
            break

        c_prev = c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    df = pd.DataFrame(historico)

    return {
        "raiz": c,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": df,
        "erro_final": erro,
        "metodo": "Falsa Posição",
    }


def falsa_posicao_illinois(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> dict:
    """
    Variante Illinois — evita estagnação dividindo f(extremo fixo) por 2
    quando o mesmo extremo é mantido duas iterações consecutivas.
    """
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos.")

    historico = []
    ultimo_lado = None  # rastreia qual extremo ficou fixo
    lado_count = 0
    c_prev = a
    convergiu = False

    for i in range(1, max_iter + 1):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        erro = abs(c - c_prev) if i > 1 else abs(b - a)

        historico.append({
            "iteração": i,
            "a": a,
            "b": b,
            "c": c,
            "f(c)": fc,
            "erro": erro,
        })

        if i > 1 and erro < tol:
            convergiu = True
            break

        c_prev = c

        if fa * fc < 0:
            lado = "b"
            if ultimo_lado == "b":
                lado_count += 1
                if lado_count >= 2:
                    fa /= 2  # Illinois
            else:
                lado_count = 1
            b, fb = c, fc
        else:
            lado = "a"
            if ultimo_lado == "a":
                lado_count += 1
                if lado_count >= 2:
                    fb /= 2  # Illinois
            else:
                lado_count = 1
            a, fa = c, fc

        ultimo_lado = lado

    return {
        "raiz": c,
        "iteracoes": i,
        "convergiu": convergiu,
        "historico": pd.DataFrame(historico),
        "erro_final": erro,
        "metodo": "Falsa Posição (Illinois)",
    }


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def f(x):
        return x**3 - x - 2

    print("=" * 55)
    print("  Falsa Posição — Clássico")
    print("=" * 55)
    r1 = falsa_posicao(f, 1.0, 2.0, tol=1e-8)
    print(f"  Raiz: {r1['raiz']:.10f}  |  Iter: {r1['iteracoes']}  |  Erro: {r1['erro_final']:.2e}")

    print()
    print("=" * 55)
    print("  Falsa Posição — Illinois (modificado)")
    print("=" * 55)
    r2 = falsa_posicao_illinois(f, 1.0, 2.0, tol=1e-8)
    print(f"  Raiz: {r2['raiz']:.10f}  |  Iter: {r2['iteracoes']}  |  Erro: {r2['erro_final']:.2e}")

    print()
    print(r1["historico"].to_string(index=False))
