"""Systematic Backtester : un framework de backtesting multi-actifs avec rigueur statistique.

Organisation du package :
    data        -> téléchargement, nettoyage et cache des prix
    strategies  -> chaque stratégie renvoie des POIDS CIBLES (rien d'autre)
    engine      -> exécution : décalage temporel, dérive des poids, rebalancement, coûts
    metrics     -> mesures de performance et de risque
    plotting    -> graphiques standards
"""

__version__ = "0.1.0"
