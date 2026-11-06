"""Stratégies.

Règle d'or : une stratégie est une fonction prix -> poids cibles.
    - entrée  : DataFrame de prix (dates x actifs)
    - sortie  : DataFrame de poids (dates x actifs), NaN tant que le signal n'existe pas
    - elle n'utilise QUE les données jusqu'à la date t pour produire le poids daté t

Elle ne gère ni le timing d'exécution, ni les coûts, ni le rebalancement :
tout cela est le travail du moteur. Ajouter une stratégie = écrire une fonction.
"""

from __future__ import annotations

import pandas as pd


def equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    """1/N sur chaque actif. Avec rebalance='never', c'est un buy & hold équipondéré."""
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def sma_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Croisement de moyennes mobiles, actif par actif.

    Chaque actif dispose d'une « case » de 1/N du portefeuille :
        - moyenne rapide > moyenne lente -> la case est investie dans l'actif ;
        - sinon -> la case reste en cash.
    """
    if fast >= slow:
        raise ValueError("La fenêtre rapide doit être plus courte que la fenêtre lente.")
    sma_fast = prices.rolling(fast, min_periods=fast).mean()
    sma_slow = prices.rolling(slow, min_periods=slow).mean()

    signal = (sma_fast > sma_slow).astype(float)
    signal = signal.where(sma_slow.notna())  # pas de signal tant que la fenêtre lente est incomplète
    return signal / prices.shape[1]


# Registre : nom lisible -> fonction. Pratique pour l'interface Streamlit plus tard.
STRATEGIES = {
    "Equal weight": equal_weight,
    "SMA crossover": sma_crossover,
}
