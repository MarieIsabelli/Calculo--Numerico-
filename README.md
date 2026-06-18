# ∿ Resolução de Equações Não Lineares

> **Cálculo Numérico**  
> Implementação e análise comparativa de métodos clássicos para encontrar raízes de funções matemáticas.

---

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Métodos Implementados](#métodos-implementados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Execução](#instalação-e-execução)
- [Como Usar](#como-usar)
- [Exemplos de Saída](#exemplos-de-saída)
- [Fundamentos Matemáticos](#fundamentos-matemáticos)
- [Comparativo de Desempenho](#comparativo-de-desempenho)
- [Tecnologias](#tecnologias)

---

## Sobre o Projeto

Aplicação Python para **resolução de equações não lineares** utilizando quatro métodos numéricos clássicos. O projeto inclui:

- ✅ Implementação modular e documentada de cada método
- ✅ Análise de convergência com visualização gráfica
- ✅ Interface web interativa com Streamlit
- ✅ Exportação de resultados em CSV e PNG
- ✅ Execução em lote via linha de comando

Desenvolvido como projeto de Cálculo Numérico, demonstrando teoria matemática, programação orientada a módulos e análise de resultados.

---

## Métodos Implementados

### 🔵 Bisseção
Divide o intervalo `[a, b]` ao meio repetidamente, mantendo o subintervalo com troca de sinal.

```
c = (a + b) / 2
```

| Propriedade | Valor |
|---|---|
| Convergência | Linear (ordem 1) |
| Garantia | Sim (Teorema de Bolzano) |
| Requisito | f(a)·f(b) < 0 |
| Iterações mínimas | ceil(log₂((b−a)/tol)) |

---

### 🟢 Falsa Posição (Regula Falsi)
Usa interpolação linear entre os extremos do intervalo — mais eficiente que a bisseção pura.

```
c = b - f(b) · (b - a) / (f(b) - f(a))
```

| Propriedade | Valor |
|---|---|
| Convergência | Superlinear (casos favoráveis) |
| Garantia | Sim (com intervalo válido) |
| Variante extra | Illinois (evita estagnação) |

---

### 🟡 Newton-Raphson
Usa a reta tangente à curva no ponto atual. O método mais rápido, com convergência quadrática.

```
x_{n+1} = x_n - f(x_n) / f'(x_n)
```

| Propriedade | Valor |
|---|---|
| Convergência | Quadrática (ordem 2) |
| Garantia | Não — depende de x₀ |
| Derivada | Analítica ou numérica automática |
| Variante extra | Raízes múltiplas (Halley) |

---

### 🔴 Secante
Aproxima a derivada por diferença finita — elimina a necessidade de calcular f' analiticamente.

```
x_{n+1} = x_n - f(x_n) · (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
```

| Propriedade | Valor |
|---|---|
| Convergência | Superlinear (ordem φ ≈ 1.618) |
| Garantia | Não — pode divergir |
| Requisito | Dois pontos iniciais x₀ e x₁ |
| Variante extra | Brent (global + rápido) |

---

## Estrutura do Projeto

```
calculo_numerico/
│
├── src/
│   ├── __init__.py          # Exporta todos os métodos
│   ├── bissecao.py          # Método da Bisseção + iterações teóricas
│   ├── falsa_posicao.py     # Falsa Posição clássico + Illinois
│   ├── newton.py            # Newton-Raphson + raízes múltiplas
│   └── secante.py           # Secante + método de Brent
│
├── dados/                   # Dados de entrada (opcional)
├── graficos/                # PNGs gerados por main.py
├── resultados/              # CSVs de iterações + resumo_geral.csv
│
├── main.py                  # CLI: roda todos os casos, gera arquivos
├── app.py                   # Interface Streamlit interativa
├── requirements.txt
└── README.md
```

---

## Instalação e Execução

### 1. Clone e instale dependências

```bash
git clone https://github.com/seu-usuario/calculo_numerico.git
cd calculo_numerico

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Interface Web (Streamlit)

```bash
streamlit run app.py
```

Acesse **http://localhost:8501** no navegador.

### 3. Linha de Comando

```bash
# Roda todos os 5 casos de teste
python main.py

# Lista os casos disponíveis
python main.py --lista

# Roda apenas o caso 0 com tolerância 1e-10
python main.py --caso 0 --tol 1e-10
```

### 4. Usar como biblioteca

```python
from src import bissecao, newton, falsa_posicao, secante
import numpy as np

def f(x):  return x**3 - x - 2
def df(x): return 3*x**2 - 1

# Bisseção
r1 = bissecao(f, a=1.0, b=2.0, tol=1e-8)
print(f"Raiz: {r1['raiz']:.10f} | Iterações: {r1['iteracoes']}")

# Newton-Raphson (derivada analítica)
r2 = newton(f, x0=1.5, df=df, tol=1e-10)
print(f"Raiz: {r2['raiz']:.10f} | Iterações: {r2['iteracoes']}")

# Ver histórico
print(r2["historico"])
```

---

## Como Usar

### Interface Streamlit

1. Selecione uma função de exemplo ou digite sua própria expressão Python
2. Configure o intervalo `[a, b]` e os pontos iniciais `x₀`, `x₁`
3. Ajuste a tolerância com o slider
4. Marque os métodos desejados
5. Clique em **Executar**
6. Explore as abas: Função · Convergência · Comparativo · Iterações · Exportar

### Expressões válidas no app

```python
x**3 - x - 2          # Polinômio
np.cos(x) - x         # Trigonométrica
np.exp(x) - 3*x       # Exponencial
np.log(x) - 1         # Logarítmica
x**2 - np.sin(x)      # Mista
```

---

## Exemplos de Saída

### Terminal (`main.py`)

```
══════════════════════════════════════════════════════════════
  Cálculo Numérico — Resolução de Equações Não Lineares
══════════════════════════════════════════════════════════════
  Tolerância : 1e-08    Casos: 5
══════════════════════════════════════════════════════════════

▶ Caso 0: Polinômio cúbico  |  f(x) = x³ − x − 2
    ✓ Bisseção            raiz=1.5213797069  iter= 27  erro=3.72e-09  (0.421 ms)
    ✓ Falsa Posição        raiz=1.5213797069  iter= 13  erro=5.14e-09  (0.198 ms)
    ✓ Newton-Raphson       raiz=1.5213797069  iter=  5  erro=2.22e-16  (0.093 ms)
    ✓ Secante              raiz=1.5213797069  iter=  7  erro=3.55e-11  (0.112 ms)
    📊 Gráfico salvo → graficos/convergencia_caso_00_...png
    💾 CSV salvo      → resultados/iteracoes_caso_00_...csv
```

### Arquivo `resultados/resumo_geral.csv`

| caso | função | método | raiz | iterações | erro_final | tempo_ms |
|---|---|---|---|---|---|---|
| Polinômio cúbico | x³−x−2 | Bisseção | 1.5213797069 | 27 | 3.72e-09 | 0.421 |
| Polinômio cúbico | x³−x−2 | Newton-Raphson | 1.5213797069 | 5 | 2.22e-16 | 0.093 |

---

## Fundamentos Matemáticos

### Critério de parada

Todos os métodos usam critério de erro absoluto:

```
|x_{n+1} - x_n| < tol
```

Com `tol = 1e-8` por padrão (configurável).

### Ordem de convergência

Para cada método, a ordem `p` satisfaz:

```
|e_{n+1}| ≈ C · |e_n|^p
```

| Método | Ordem p | Tipo |
|---|---|---|
| Bisseção | 1.0 | Linear |
| Falsa Posição | 1.0–2.0 | Sublinear/Linear |
| Newton-Raphson | 2.0 | Quadrática |
| Secante | 1.618 (φ) | Superlinear |

### Teorema de Bolzano

Para Bisseção e Falsa Posição, é necessário que:

```
f(a) · f(b) < 0
```

Isso garante a existência de pelo menos uma raiz em `(a, b)` pela continuidade de f.

---

## Comparativo de Desempenho

Função `f(x) = x³ − x − 2`, tolerância `1e-8`:

| Método | Iterações | Velocidade | Robustez | Precisa de f'? |
|---|---|---|---|---|
| Bisseção | ~27 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Não |
| Falsa Posição | ~13 | ⭐⭐⭐ | ⭐⭐⭐⭐ | Não |
| Newton-Raphson | ~5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Sim (ou numérica) |
| Secante | ~7 | ⭐⭐⭐⭐ | ⭐⭐⭐ | Não |

> **Recomendação geral:** Newton-Raphson quando f' é conhecida; Secante como alternativa sem derivada; Bisseção quando robustez é prioritária.

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| `numpy` | Operações vetoriais e funções matemáticas |
| `pandas` | Tabelas de iterações e exportação CSV |
| `matplotlib` | Gráficos de função e convergência |
| `streamlit` | Interface web interativa |
| `scipy` | Referência (método de Brent) |

---
---

<div align="center">
  <sub>Desenvolvido por Marie  · Cálculo Numérico</sub>
</div>
