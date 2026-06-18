# src/__init__.py
from .bissecao      import bissecao, bissecao_teorico_iter
from .falsa_posicao import falsa_posicao, falsa_posicao_illinois
from .newton        import newton, newton_multiplas_raizes
from .secante       import secante, secante_brent

__all__ = [
    "bissecao", "bissecao_teorico_iter",
    "falsa_posicao", "falsa_posicao_illinois",
    "newton", "newton_multiplas_raizes",
    "secante", "secante_brent",
]
