"""
app.py — Dashboard Streamlit: Resolução de Equações Não Lineares
================================================================
Execute com:
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import io
import time

from bissecao     import bissecao
from falsa_posicao import falsa_posicao
from newton       import newton
from secante      import secante

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cálculo Numérico",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo escuro geral */
    .stApp { background-color: #020817; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #080d1a; border-right: 1px solid #0f172a; }
    .block-container { padding-top: 1.5rem; }

    /* Métricas */
    [data-testid="metric-container"] {
        background: #0a0f1e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: #475569 !important; font-size: 11px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 22px !important; font-family: 'JetBrains Mono', monospace; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background: #080d1a; border-bottom: 1px solid #0f172a; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #475569; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { background: #0f172a !important; color: #a5b4fc !important; }

    /* Botão primário */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none; border-radius: 8px; font-weight: 600;
        padding: 0.5rem 2rem; font-size: 15px;
    }

    /* Cabeçalho de seção */
    .secao-header {
        font-size: 13px; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: .1em;
        margin: 16px 0 8px;
    }

    /* Badge de status */
    .badge-ok  { background:#10b98120; color:#10b981; padding:3px 10px; border-radius:20px; font-size:11px; }
    .badge-err { background:#ef444420; color:#ef4444; padding:3px 10px; border-radius:20px; font-size:11px; }

    /* Destaque de raiz */
    .raiz-box {
        background: #0f172a; border: 1px solid #1e293b; border-left: 3px solid #6366f1;
        border-radius: 8px; padding: 12px 18px; font-family: monospace;
        font-size: 15px; color: #a5b4fc; margin: 8px 0;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li { color: #94a3b8; }
    code { background: #0f172a; color: #a5b4fc; border-radius: 4px; padding: 2px 6px; }
</style>
""", unsafe_allow_html=True)

# ── Paleta ────────────────────────────────────────────────────────────────────
CORES = {
    "Bisseção":       "#6366f1",
    "Falsa Posição":  "#10b981",
    "Newton-Raphson": "#f59e0b",
    "Secante":        "#ef4444",
}

PRESETS = [
    {"label": "x³ − x − 2",     "expr": "x**3 - x - 2",     "a": 1.0, "b": 2.0, "x0": 1.5, "x1": 2.0},
    {"label": "cos(x) − x",     "expr": "np.cos(x) - x",    "a": 0.0, "b": 1.0, "x0": 0.5, "x1": 1.0},
    {"label": "eˣ − 3x",        "expr": "np.exp(x) - 3*x",  "a": 0.0, "b": 2.0, "x0": 0.5, "x1": 1.5},
    {"label": "x² − 2",         "expr": "x**2 - 2",          "a": 1.0, "b": 2.0, "x0": 1.5, "x1": 2.0},
    {"label": "sin(x) − x/2",   "expr": "np.sin(x) - x/2",  "a": 1.5, "b": 3.0, "x0": 2.0, "x1": 2.5},
    {"label": "Personalizada",   "expr": "",                  "a": 0.0, "b": 2.0, "x0": 1.0, "x1": 1.5},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_eval(expr: str):
    """Retorna uma função f(x) a partir de uma expressão Python."""
    def f(x):
        try:
            return eval(expr, {"x": x, "np": np, **{k: getattr(np, k)
                        for k in ["sin","cos","tan","exp","log","sqrt","pi","e"]}})
        except Exception:
            return np.nan
    return f


def derivada_numerica(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def rodar_metodos(f, a, b, x0, x1, tol, metodos_ativos):
    resultados = {}
    df = lambda x: derivada_numerica(f, x)

    runners = {
        "Bisseção":       lambda: bissecao(f, a, b, tol=tol),
        "Falsa Posição":  lambda: falsa_posicao(f, a, b, tol=tol),
        "Newton-Raphson": lambda: newton(f, x0, df=df, tol=tol),
        "Secante":        lambda: secante(f, x0, x1, tol=tol),
    }

    for nome, runner in runners.items():
        if not metodos_ativos.get(nome, False):
            continue
        t0 = time.perf_counter()
        try:
            res = runner()
            res["tempo_ms"] = (time.perf_counter() - t0) * 1000
            resultados[nome] = res
        except Exception as e:
            resultados[nome] = {"error": str(e), "metodo": nome}

    return resultados


def fig_funcao(f, a, b, resultados, expr):
    """Gráfico da função com as raízes marcadas."""
    lo, hi = min(a, b) - 1.2, max(a, b) + 1.2
    xs = np.linspace(lo, hi, 800)
    ys = np.vectorize(f)(xs)
    mask = np.abs(ys) < 50
    xs_plot, ys_plot = xs[mask], ys[mask]

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#0a0f1e")
    ax.set_facecolor("#0a0f1e")
    ax.plot(xs_plot, ys_plot, color="#6366f1", lw=2.4, label=f"f(x) = {expr}")
    ax.axhline(0, color="#334155", lw=1.2)
    ax.axvspan(a, b, alpha=0.07, color="#6366f1", label=f"Intervalo [{a}, {b}]")

    raizes_plotadas = set()
    for nome, cor in CORES.items():
        r = resultados.get(nome, {})
        if "raiz" in r:
            rv = round(r["raiz"], 7)
            if rv not in raizes_plotadas:
                ax.scatter([r["raiz"]], [f(r["raiz"])], color=cor, s=100,
                           zorder=5, label=f"{nome}: {r['raiz']:.7f}")
                raizes_plotadas.add(rv)

    ax.tick_params(colors="#475569", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e293b")
    ax.grid(True, color="#0f172a", lw=0.9)
    ax.set_xlabel("x", color="#64748b", fontsize=9)
    ax.set_ylabel("f(x)", color="#64748b", fontsize=9)
    ax.legend(fontsize=7.5, facecolor="#0a0f1e", edgecolor="#1e293b",
              labelcolor="#94a3b8", loc="best")
    fig.tight_layout()
    return fig


def fig_convergencia(resultados):
    """Curvas de convergência em escala log."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#0a0f1e")
    ax.set_facecolor("#0a0f1e")
    estilos = ["-", "--", "-.", ":"]

    for (nome, cor), ls in zip(CORES.items(), estilos):
        r = resultados.get(nome, {})
        if "historico" not in r: continue
        hist = r["historico"]
        col_e = [c for c in hist.columns if "erro" in c.lower()]
        if not col_e: continue
        erros = hist[col_e[0]].values.astype(float)
        erros = np.where(erros > 0, erros, np.nan)
        ax.semilogy(range(1, len(erros)+1), erros, color=cor, lw=2, ls=ls,
                    marker="o", ms=4, label=f"{nome} ({r.get('iteracoes','?')} iter.)")

    ax.set_xlabel("Iteração", color="#64748b", fontsize=9)
    ax.set_ylabel("Erro absoluto (escala log)", color="#64748b", fontsize=9)
    ax.tick_params(colors="#475569", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e293b")
    ax.grid(True, which="both", color="#0f172a", lw=0.8)
    ax.legend(fontsize=8, facecolor="#0a0f1e", edgecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig


def fig_barras(resultados):
    """Barras comparativas de iterações e tempo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), facecolor="#0a0f1e")

    nomes = [n for n in CORES if "iteracoes" in resultados.get(n, {})]
    iters = [resultados[n]["iteracoes"] for n in nomes]
    tempos = [resultados[n].get("tempo_ms", 0) for n in nomes]
    cores  = [CORES[n] for n in nomes]

    for ax, vals, ylabel, title in [
        (ax1, iters,  "Iterações",  "Iterações necessárias"),
        (ax2, tempos, "Tempo (ms)", "Tempo de execução (ms)"),
    ]:
        ax.set_facecolor("#0a0f1e")
        bars = ax.bar(nomes, vals, color=cores, edgecolor="#0a0f1e", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.3f}" if isinstance(v, float) else str(v),
                    ha="center", va="bottom", color="#e2e8f0", fontsize=8)
        ax.set_ylabel(ylabel, color="#64748b", fontsize=8)
        ax.set_title(title, color="#94a3b8", fontsize=9, pad=6)
        ax.tick_params(colors="#475569", labelsize=7.5)
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        ax.grid(True, axis="y", color="#0f172a", lw=0.8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7.5)

    fig.set_facecolor("#0a0f1e")
    fig.tight_layout()
    return fig


def gerar_csv(resultados) -> bytes:
    """Gera bytes do CSV com histórico de iterações."""
    frames = []
    for nome, r in resultados.items():
        if "historico" not in r: continue
        h = r["historico"].copy()
        h.insert(0, "método", nome)
        frames.append(h)
    if not frames:
        return b""
    df = pd.concat(frames, ignore_index=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.10f")
    return buf.getvalue().encode()


def gerar_fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ∿ &nbsp;Cálculo Numérico", unsafe_allow_html=True)
    st.caption("Resolução de Equações Não Lineares")
    st.divider()

    # Preset
    st.markdown('<div class="secao-header">Função f(x)</div>', unsafe_allow_html=True)
    preset_idx = st.selectbox(
        "Exemplo predefinido",
        options=range(len(PRESETS)),
        format_func=lambda i: PRESETS[i]["label"],
        label_visibility="collapsed",
    )
    p = PRESETS[preset_idx]

    if p["label"] == "Personalizada":
        expr_input = st.text_input("Expressão Python", value="x**3 - x - 2",
                                   help="Use np.sin, np.cos, np.exp, **")
    else:
        expr_input = p["expr"]
        st.code(f"f(x) = {p['label']}", language="python")

    st.markdown('<div class="secao-header">Parâmetros</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    a   = col1.number_input("a", value=float(p["a"]),  format="%.4f")
    b   = col2.number_input("b", value=float(p["b"]),  format="%.4f")
    x0  = col1.number_input("x₀", value=float(p["x0"]), format="%.4f")
    x1  = col2.number_input("x₁", value=float(p["x1"]), format="%.4f",
                             help="Segundo ponto para o Método da Secante")

    tol_exp = st.slider("Tolerância (10⁻ⁿ)", min_value=2, max_value=12, value=8)
    tol = 10 ** (-tol_exp)
    st.caption(f"tol = 1e-{tol_exp} = {tol:.0e}")

    st.markdown('<div class="secao-header">Métodos ativos</div>', unsafe_allow_html=True)
    metodos_ativos = {}
    for nome, cor in CORES.items():
        metodos_ativos[nome] = st.checkbox(
            nome, value=True,
            help={"Bisseção": "Requer intervalo [a,b] com troca de sinal",
                  "Falsa Posição": "Interpolação linear sobre [a,b]",
                  "Newton-Raphson": "Usa derivada numérica automaticamente",
                  "Secante": "Requer dois pontos iniciais x₀ e x₁"}[nome]
        )

    st.divider()
    executar = st.button("▶  Executar", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESSÃO
# ══════════════════════════════════════════════════════════════════════════════

if "resultados" not in st.session_state:
    st.session_state.resultados = {}
if "executado" not in st.session_state:
    st.session_state.executado = False

if executar:
    f = safe_eval(expr_input)
    with st.spinner("Calculando..."):
        st.session_state.resultados = rodar_metodos(
            f, a, b, x0, x1, tol, metodos_ativos
        )
        st.session_state.executado = True
        st.session_state.f = f
        st.session_state.expr = expr_input
        st.session_state.a, st.session_state.b = a, b


# ══════════════════════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.title("Resolução de Equações Não Lineares")
st.caption("Bisseção · Falsa Posição · Newton-Raphson · Secante — análise e comparação de convergência")

if not st.session_state.executado:
    # ── Tela inicial ─────────────────────────────────────────────────────────
    st.info("👈  Configure os parâmetros na barra lateral e clique em **Executar**.")
    st.markdown("---")

    col_a, col_b, col_c, col_d = st.columns(4)
    infos = [
        ("🔵", "Bisseção", "Divide o intervalo ao meio a cada iteração. Garantia de convergência. Ordem 1 (linear)."),
        ("🟢", "Falsa Posição", "Interpola linearmente entre a e b. Mais rápida que bisseção em geral. Pode estagnar."),
        ("🟡", "Newton-Raphson", "Usa a reta tangente. Convergência quadrática. Pode divergir longe da raiz."),
        ("🔴", "Secante", "Aproxima a derivada por diferença finita. Ordem φ ≈ 1.618. Não requer f'."),
    ]
    for col, (ico, nome, desc) in zip([col_a, col_b, col_c, col_d], infos):
        with col:
            st.markdown(f"### {ico} {nome}")
            st.markdown(f"<p style='font-size:13px'>{desc}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Fórmulas dos métodos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bisseção / Falsa Posição**")
        st.latex(r"c = \frac{a + b}{2} \quad \text{(bisseção)}")
        st.latex(r"c = b - f(b)\frac{b - a}{f(b) - f(a)} \quad \text{(falsa posição)}")
    with col2:
        st.markdown("**Newton-Raphson / Secante**")
        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
        st.latex(r"x_{n+1} = x_n - f(x_n)\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}")

else:
    resultados = st.session_state.resultados
    f          = st.session_state.f
    expr       = st.session_state.expr
    _a, _b     = st.session_state.a, st.session_state.b

    # ── Métricas de topo ─────────────────────────────────────────────────────
    metodos_ok = {n: r for n, r in resultados.items() if "raiz" in r}
    metodos_err = {n: r for n, r in resultados.items() if "error" in r}

    if metodos_ok:
        cols = st.columns(len(metodos_ok))
        for col, (nome, r) in zip(cols, metodos_ok.items()):
            with col:
                st.metric(
                    label=nome,
                    value=f"{r['raiz']:.8f}",
                    delta=f"{r['iteracoes']} iterações · {r['erro_final']:.1e}",
                )

    for nome, r in metodos_err.items():
        st.error(f"**{nome}**: {r['error']}")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Função", "📉 Convergência", "📊 Comparativo", "📋 Iterações", "⬇ Exportar"
    ])

    with tab1:
        st.subheader("Gráfico da Função")
        fig1 = fig_funcao(f, _a, _b, resultados, expr)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown("**Valores no intervalo**")
        c1, c2, c3, c4 = st.columns(4)
        fa = f(_a); fb = f(_b)
        c1.metric("f(a)", f"{fa:.6f}", delta="⚠ mesmo sinal" if fa*fb > 0 else None,
                  delta_color="inverse" if fa*fb > 0 else "normal")
        c2.metric("f(b)", f"{fb:.6f}")
        c3.metric("f(a)·f(b)", f"{fa*fb:.4e}",
                  delta="✗ sem garantia" if fa*fb > 0 else "✓ Bolzano OK",
                  delta_color="inverse" if fa*fb > 0 else "normal")
        c4.metric("Largura [a,b]", f"{abs(_b-_a):.4f}")

    with tab2:
        st.subheader("Curvas de Convergência")
        fig2 = fig_convergencia(resultados)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        if metodos_ok:
            melhor = min(metodos_ok, key=lambda n: metodos_ok[n]["iteracoes"])
            pior   = max(metodos_ok, key=lambda n: metodos_ok[n]["iteracoes"])
            st.info(
                f"🏆 **Mais rápido**: {melhor} com {metodos_ok[melhor]['iteracoes']} iterações.  "
                f"  🐢 **Mais lento**: {pior} com {metodos_ok[pior]['iteracoes']} iterações."
            )

    with tab3:
        st.subheader("Comparação Visual")
        if metodos_ok:
            fig3 = fig_barras(resultados)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            st.markdown("**Tabela resumo**")
            rows = []
            for nome, r in resultados.items():
                rows.append({
                    "Método":       nome,
                    "Raiz":         f"{r.get('raiz', 'ERRO'):.10f}" if "raiz" in r else "ERRO",
                    "Iterações":    r.get("iteracoes", "—"),
                    "Erro final":   f"{r.get('erro_final', 0):.2e}" if "raiz" in r else "—",
                    "Tempo (ms)":   f"{r.get('tempo_ms', 0):.4f}" if "raiz" in r else "—",
                    "Convergiu":    "✓" if r.get("convergiu") else ("✗ ERRO" if "error" in r else "~"),
                })
            st.dataframe(
                pd.DataFrame(rows).set_index("Método"),
                use_container_width=True,
            )

    with tab4:
        st.subheader("Histórico de Iterações")
        metodo_sel = st.selectbox(
            "Selecione o método",
            options=[n for n in CORES if n in resultados and "historico" in resultados[n]],
        )
        if metodo_sel:
            r = resultados[metodo_sel]
            hist = r["historico"]
            st.dataframe(
                hist.style.format(precision=10).background_gradient(
                    subset=[c for c in hist.columns if "erro" in c.lower()],
                    cmap="RdYlGn_r"
                ),
                use_container_width=True,
                height=420,
            )

            # Mini gráfico do erro desse método
            col_e = [c for c in hist.columns if "erro" in c.lower()]
            if col_e:
                erros = hist[col_e[0]].values.astype(float)
                erros = np.where(erros > 0, erros, np.nan)
                fig_mini, ax = plt.subplots(figsize=(7, 2.8), facecolor="#0a0f1e")
                ax.set_facecolor("#0a0f1e")
                ax.semilogy(range(1, len(erros)+1), erros,
                            color=CORES[metodo_sel], lw=2, marker="o", ms=5)
                ax.set_xlabel("Iteração", color="#64748b", fontsize=9)
                ax.set_ylabel("Erro (log)", color="#64748b", fontsize=9)
                ax.tick_params(colors="#475569", labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#1e293b")
                ax.grid(True, which="both", color="#0f172a")
                ax.set_title(f"Convergência — {metodo_sel}", color="#94a3b8", fontsize=9)
                fig_mini.tight_layout()
                st.pyplot(fig_mini, use_container_width=True)
                plt.close(fig_mini)

    with tab5:
        st.subheader("Exportar Resultados")
        c1, c2, c3 = st.columns(3)

        # CSV de iterações
        csv_bytes = gerar_csv(resultados)
        c1.download_button(
            "⬇ Histórico CSV",
            data=csv_bytes,
            file_name="iteracoes_convergencia.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # PNG da função
        fig_fn = fig_funcao(f, _a, _b, resultados, expr)
        c2.download_button(
            "⬇ Gráfico da Função (.png)",
            data=gerar_fig_png(fig_fn),
            file_name="grafico_funcao.png",
            mime="image/png",
            use_container_width=True,
        )
        plt.close(fig_fn)

        # PNG da convergência
        fig_cv = fig_convergencia(resultados)
        c3.download_button(
            "⬇ Convergência (.png)",
            data=gerar_fig_png(fig_cv),
            file_name="grafico_convergencia.png",
            mime="image/png",
            use_container_width=True,
        )
        plt.close(fig_cv)

        if csv_bytes:
            st.markdown("**Pré-visualização do CSV**")
            preview = pd.read_csv(io.StringIO(csv_bytes.decode()))
            st.dataframe(preview.head(20), use_container_width=True)
