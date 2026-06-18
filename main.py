"""
main.py — Resolução de Equações Não Lineares e Análise de Convergência
========================================================================
Executa todos os quatro métodos numéricos em um conjunto de funções de teste,
gera gráficos comparativos, tabelas de convergência e exporta os resultados
em CSV. Tudo configurável via dicionário CASOS abaixo.

Uso:
    python main.py              → roda todos os casos
    python main.py --caso 0    → roda apenas o caso de índice 0
    python main.py --lista     → lista os casos disponíveis
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator

# Garante que os módulos locais sejam encontrados
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bissecao import bissecao, bissecao_teorico_iter
from falsa_posicao import falsa_posicao, falsa_posicao_illinois
from newton import newton
from secante import secante

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # sem display — salva direto em arquivo


# ── Pasta de saída ────────────────────────────────────────────────────────────
DIR_GRAFICOS  = os.path.join(os.path.dirname(__file__), "graficos")
DIR_RESULTADOS = os.path.join(os.path.dirname(__file__), "resultados")
os.makedirs(DIR_GRAFICOS,  exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


# ── Casos de teste ────────────────────────────────────────────────────────────
CASOS = [
    {
        "nome": "Polinômio cúbico",
        "expr": "x³ − x − 2",
        "f":    lambda x: x**3 - x - 2,
        "df":   lambda x: 3*x**2 - 1,
        "a": 1.0, "b": 2.0, "x0": 1.5, "x1": 2.0,
        "xmin": -2.5, "xmax": 3.0,
    },
    {
        "nome": "Cosseno vs identidade",
        "expr": "cos(x) − x",
        "f":    lambda x: np.cos(x) - x,
        "df":   lambda x: -np.sin(x) - 1,
        "a": 0.0, "b": 1.0, "x0": 0.5, "x1": 1.0,
        "xmin": -0.5, "xmax": 2.0,
    },
    {
        "nome": "Exponencial vs linear",
        "expr": "eˣ − 3x",
        "f":    lambda x: np.exp(x) - 3*x,
        "df":   lambda x: np.exp(x) - 3,
        "a": 0.0, "b": 2.0, "x0": 0.5, "x1": 1.5,
        "xmin": -0.5, "xmax": 2.5,
    },
    {
        "nome": "Raiz quadrada de 2",
        "expr": "x² − 2",
        "f":    lambda x: x**2 - 2,
        "df":   lambda x: 2*x,
        "a": 1.0, "b": 2.0, "x0": 1.5, "x1": 2.0,
        "xmin": -0.5, "xmax": 2.5,
    },
    {
        "nome": "Seno vs metade",
        "expr": "sin(x) − x/2",
        "f":    lambda x: np.sin(x) - x/2,
        "df":   lambda x: np.cos(x) - 0.5,
        "a": 1.5, "b": 3.0, "x0": 2.0, "x1": 2.5,
        "xmin": -0.5, "xmax": 4.0,
    },
]

METODOS = ["Bisseção", "Falsa Posição", "Newton-Raphson", "Secante"]
CORES   = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
ESTILO  = ["-", "--", "-.", ":"]


# ── Executor dos métodos ──────────────────────────────────────────────────────

def rodar_metodos(caso: dict, tol: float = 1e-8) -> dict:
    """Executa os quatro métodos e retorna um dicionário com resultados."""
    f, df = caso["f"], caso["df"]
    resultados = {}

    runners = [
        ("Bisseção",       lambda: bissecao(f, caso["a"], caso["b"], tol=tol)),
        ("Falsa Posição",  lambda: falsa_posicao(f, caso["a"], caso["b"], tol=tol)),
        ("Newton-Raphson", lambda: newton(f, caso["x0"], df=df, tol=tol)),
        ("Secante",        lambda: secante(f, caso["x0"], caso["x1"], tol=tol)),
    ]

    for nome, runner in runners:
        t0 = time.perf_counter()
        try:
            res = runner()
            res["tempo_ms"] = (time.perf_counter() - t0) * 1000
            resultados[nome] = res
        except Exception as e:
            resultados[nome] = {"error": str(e), "metodo": nome}

    return resultados


# ── Geração de gráficos ────────────────────────────────────────────────────────

def plotar(caso: dict, resultados: dict, idx: int):
    """Gera figura com 3 painéis: função, convergência e tabela-resumo."""
    f = caso["f"]
    fig = plt.figure(figsize=(16, 9), facecolor="#020817")
    fig.suptitle(
        f"Análise de Convergência  ·  {caso['expr']}",
        color="#e2e8f0", fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.10)

    # ── Painel 1: Gráfico da função ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0a0f1e")
    xs = np.linspace(caso["xmin"], caso["xmax"], 600)
    ys = np.vectorize(f)(xs)
    ax1.plot(xs, ys, color="#6366f1", lw=2.2, label=f"f(x) = {caso['expr']}")
    ax1.axhline(0, color="#334155", lw=1)
    ax1.axvline(0, color="#334155", lw=1, ls="--")

    # Marcar raízes encontradas
    raizes_plotadas = set()
    for nome, cor in zip(METODOS, CORES):
        r = resultados.get(nome, {})
        if "raiz" in r:
            raiz = round(r["raiz"], 6)
            if raiz not in raizes_plotadas:
                ax1.scatter([r["raiz"]], [f(r["raiz"])], color=cor, s=80, zorder=5,
                            label=f"{nome}: x≈{r['raiz']:.6f}")
                raizes_plotadas.add(raiz)

    ax1.axvspan(caso["a"], caso["b"], alpha=0.07, color="#6366f1")
    ax1.set_title("Função e raízes encontradas", color="#94a3b8", fontsize=10, pad=8)
    ax1.tick_params(colors="#475569", labelsize=8)
    for spine in ax1.spines.values(): spine.set_color("#1e293b")
    ax1.grid(True, color="#0f172a", lw=0.8)
    ax1.legend(fontsize=7.5, facecolor="#0f172a", edgecolor="#1e293b",
               labelcolor="#94a3b8", loc="upper left")
    ax1.set_xlabel("x", color="#475569", fontsize=9)
    ax1.set_ylabel("f(x)", color="#475569", fontsize=9)

    # ── Painel 2: Curvas de convergência ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0a0f1e")

    for nome, cor, ls in zip(METODOS, CORES, ESTILO):
        r = resultados.get(nome, {})
        if "historico" in r and r["historico"] is not None:
            hist = r["historico"]
            col_erro = [c for c in hist.columns if "erro" in c.lower()]
            if col_erro:
                erros = hist[col_erro[0]].values
                erros = np.where(erros > 0, erros, np.nan)
                ax2.semilogy(range(1, len(erros) + 1), erros,
                             color=cor, lw=2, ls=ls, marker="o", ms=4,
                             label=f"{nome} ({r.get('iteracoes', '?')} iter.)")

    ax2.set_title("Convergência do erro por iteração", color="#94a3b8", fontsize=10, pad=8)
    ax2.set_xlabel("Iteração", color="#475569", fontsize=9)
    ax2.set_ylabel("Erro absoluto (log)", color="#475569", fontsize=9)
    ax2.tick_params(colors="#475569", labelsize=8)
    for spine in ax2.spines.values(): spine.set_color("#1e293b")
    ax2.grid(True, which="both", color="#0f172a", lw=0.8)
    ax2.yaxis.set_minor_locator(LogLocator(subs="all", numticks=8))
    ax2.legend(fontsize=7.5, facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#94a3b8")

    # ── Painel 3: Barras — número de iterações ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0a0f1e")

    nomes_ok = [n for n in METODOS if "iteracoes" in resultados.get(n, {})]
    iters_ok = [resultados[n]["iteracoes"] for n in nomes_ok]
    cores_ok  = [CORES[METODOS.index(n)] for n in nomes_ok]

    bars = ax3.bar(nomes_ok, iters_ok, color=cores_ok, edgecolor="#0a0f1e", width=0.5)
    for bar, val in zip(bars, iters_ok):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha="center", va="bottom", color="#e2e8f0", fontsize=9)

    ax3.set_title("Iterações necessárias", color="#94a3b8", fontsize=10, pad=8)
    ax3.set_ylabel("Nº de iterações", color="#475569", fontsize=9)
    ax3.tick_params(colors="#475569", labelsize=8)
    for spine in ax3.spines.values(): spine.set_color("#1e293b")
    ax3.grid(True, axis="y", color="#0f172a", lw=0.8)
    plt.setp(ax3.get_xticklabels(), rotation=15, ha="right", fontsize=8)

    # ── Painel 4: Tabela resumo ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0a0f1e")
    ax4.axis("off")

    linhas, cores_linhas = [], []
    for nome, cor in zip(METODOS, CORES):
        r = resultados.get(nome, {})
        if "error" in r:
            linhas.append([nome, "ERRO", r["error"][:22], "—"])
            cores_linhas.append("#ef4444")
        elif "raiz" in r:
            linhas.append([
                nome,
                f"{r['raiz']:.8f}",
                f"{r['iteracoes']}",
                f"{r['erro_final']:.2e}",
            ])
            cores_linhas.append(cor)

    tbl = ax4.table(
        cellText=linhas,
        colLabels=["Método", "Raiz", "Iter.", "Erro"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#0f172a")
        if r == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color="#94a3b8", fontweight="bold")
        else:
            cell.set_facecolor("#080d1a")
            cell.set_text_props(color=cores_linhas[r - 1] if r <= len(cores_linhas) else "#e2e8f0")

    ax4.set_title("Tabela comparativa", color="#94a3b8", fontsize=10, pad=8)

    # ── Salvar ────────────────────────────────────────────────────────────────
    nome_arquivo = f"convergencia_caso_{idx:02d}_{caso['nome'].replace(' ', '_').lower()}.png"
    caminho = os.path.join(DIR_GRAFICOS, nome_arquivo)
    fig.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    📊 Gráfico salvo → graficos/{nome_arquivo}")
    return caminho


# ── Exportação CSV ────────────────────────────────────────────────────────────

def exportar_csv(caso: dict, resultados: dict, idx: int):
    """Gera um CSV com o histórico de todas as iterações de todos os métodos."""
    linhas = []
    for nome, r in resultados.items():
        if "historico" not in r or r["historico"] is None:
            continue
        hist = r["historico"].copy()
        hist.insert(0, "método", nome)
        hist.insert(1, "caso", caso["nome"])
        linhas.append(hist)

    if not linhas:
        return

    df_full = pd.concat(linhas, ignore_index=True)
    nome_csv = f"iteracoes_caso_{idx:02d}_{caso['nome'].replace(' ', '_').lower()}.csv"
    caminho  = os.path.join(DIR_RESULTADOS, nome_csv)
    df_full.to_csv(caminho, index=False, float_format="%.10f")
    print(f"    💾 CSV salvo      → resultados/{nome_csv}")
    return caminho


def exportar_resumo_geral(todos: list):
    """Gera resumo_geral.csv com uma linha por método/caso."""
    linhas = []
    for caso, resultados in todos:
        for nome, r in resultados.items():
            linhas.append({
                "caso":    caso["nome"],
                "função":  caso["expr"],
                "método":  nome,
                "raiz":    r.get("raiz", "ERRO"),
                "iterações": r.get("iteracoes", "—"),
                "erro_final": r.get("erro_final", "—"),
                "tempo_ms":   r.get("tempo_ms", "—"),
                "convergiu":  r.get("convergiu", False),
            })

    df = pd.DataFrame(linhas)
    caminho = os.path.join(DIR_RESULTADOS, "resumo_geral.csv")
    df.to_csv(caminho, index=False)
    print(f"\n  📋 Resumo geral → resultados/resumo_geral.csv")
    return caminho


# ── CLI e Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cálculo Numérico — Raízes de Equações")
    p.add_argument("--caso",  type=int, default=None, help="Índice do caso (0 a 4)")
    p.add_argument("--lista", action="store_true",    help="Lista os casos disponíveis")
    p.add_argument("--tol",   type=float, default=1e-8, help="Tolerância (padrão: 1e-8)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.lista:
        print("\nCasos disponíveis:\n")
        for i, c in enumerate(CASOS):
            print(f"  [{i}] {c['nome']:25s}  f(x) = {c['expr']}")
        print()
        return

    casos_rodar = [CASOS[args.caso]] if args.caso is not None else CASOS
    indices     = [args.caso] if args.caso is not None else list(range(len(CASOS)))

    print("\n" + "═" * 62)
    print("  Cálculo Numérico — Resolução de Equações Não Lineares")
    print("═" * 62)
    print(f"  Tolerância : {args.tol}")
    print(f"  Casos      : {len(casos_rodar)}")
    print("═" * 62 + "\n")

    todos = []

    for idx, caso in zip(indices, casos_rodar):
        print(f"▶ Caso {idx}: {caso['nome']}  |  f(x) = {caso['expr']}")
        resultados = rodar_metodos(caso, tol=args.tol)

        # Exibir resumo no terminal
        for nome, r in resultados.items():
            if "error" in r:
                print(f"    ✗ {nome:20s} — ERRO: {r['error']}")
            else:
                flag = "✓" if r.get("convergiu") else "~"
                print(f"    {flag} {nome:20s} raiz={r['raiz']:.10f}  "
                      f"iter={r['iteracoes']:3d}  erro={r['erro_final']:.2e}  "
                      f"({r['tempo_ms']:.3f} ms)")

        plotar(caso, resultados, idx)
        exportar_csv(caso, resultados, idx)
        todos.append((caso, resultados))
        print()

    exportar_resumo_geral(todos)

    print("\n" + "═" * 62)
    print("  Concluído! Verifique as pastas graficos/ e resultados/")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
