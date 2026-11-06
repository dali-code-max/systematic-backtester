"""Moteur de backtest.

Convention temporelle (le point le plus important de tout le projet) :

    * Le poids cible daté t est calculé avec l'information disponible à la clôture de t.
    * Le rebalancement s'exécute à la clôture de t.
    * Le portefeuille ainsi constitué ne gagne (ou perd) qu'à partir du jour t+1.

Le moteur applique donc automatiquement un décalage d'un jour : une stratégie
ne peut PAS profiter du rendement du jour où son signal est calculé
(pas de look-ahead bias).

Entre deux dates de rebalancement, les poids DÉRIVENT avec les prix,
comme dans un vrai portefeuille. La part non investie est du cash (rendement nul).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REBALANCE_FREQUENCIES = {"D", "W", "M", "Q", "never"}


@dataclass
class BacktestResult:
    """Tout ce qu'il faut pour analyser un backtest."""

    name: str
    returns: pd.Series          # rendements quotidiens nets de coûts
    gross_returns: pd.Series    # rendements quotidiens avant coûts
    weights: pd.DataFrame       # poids détenus pendant chaque journée
    turnover: pd.Series         # somme des |variations de poids| échangées chaque jour
    costs: pd.Series            # coûts de transaction (en fraction du portefeuille)
    params: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        """Valeur du portefeuille, base 1 au départ."""
        return (1 + self.returns).cumprod()


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Dernier jour de bourse de chaque période (semaine, mois, trimestre)."""
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"Fréquence inconnue '{frequency}', choisir parmi {REBALANCE_FREQUENCIES}")
    if frequency == "D":
        return index
    if frequency == "never":
        return index[:1]
    dates = index.to_series()
    return pd.DatetimeIndex(dates.groupby(index.to_period(frequency)).last().values)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance: str = "M",
    cost_bps: float = 5.0,
    start: str | pd.Timestamp | None = None,
    name: str = "strategy",
) -> BacktestResult:
    """Simule une stratégie à partir de ses poids cibles.

    prices         : prix ajustés (dates x actifs)
    target_weights : poids cibles (dates x actifs) ; NaN = pas encore de signal
    rebalance      : 'D', 'W', 'M', 'Q' ou 'never' (achat initial puis conservation)
    cost_bps       : coût de transaction en points de base du montant échangé
    start          : date de début du trading ; les signaux peuvent utiliser
                     l'historique antérieur (fenêtres de calcul), mais on ne trade pas avant
    """
    prices = prices.sort_index()
    if start is not None:
        start = pd.Timestamp(start)

    # Rendements simples des actifs (le premier jour : 0)
    asset_returns = (prices / prices.shift(1) - 1).fillna(0.0)
    targets = target_weights.reindex(index=prices.index, columns=prices.columns)

    trading_index = prices.index if start is None else prices.index[prices.index >= start]
    if len(trading_index) == 0:
        raise ValueError("Aucune date de trading après 'start'.")

    # Dates où l'on a le droit de trader : calendrier de rebalancement, à partir de 'start'.
    # Pour 'never', le seul rebalancement est la première date où un signal existe.
    if rebalance == "never":
        has_signal = targets.loc[trading_index].notna().any(axis=1)
        allowed = set(has_signal[has_signal].index[:1])
    else:
        allowed = set(rebalance_dates(trading_index, rebalance))

    rets = asset_returns.loc[trading_index].to_numpy()
    tgt = targets.loc[trading_index].to_numpy()
    n_days, n_assets = rets.shape

    held = np.zeros((n_days, n_assets))  # poids détenus pendant la journée t
    gross = np.zeros(n_days)
    turnover = np.zeros(n_days)
    w = np.zeros(n_assets)               # poids courants (cash = 1 - somme)

    for t, date in enumerate(trading_index):
        # 1) Pendant la journée t, on détient les poids fixés à la clôture de t-1.
        held[t] = w
        port_ret = float(w @ rets[t])  # la part en cash rapporte 0
        gross[t] = port_ret

        # 2) Dérive des poids avec les prix jusqu'à la clôture.
        if abs(1 + port_ret) > 1e-12:
            w = w * (1 + rets[t]) / (1 + port_ret)

        # 3) À la clôture de t : rebalancement éventuel vers la cible (connue à t).
        target_row = tgt[t]
        if date in allowed and not np.all(np.isnan(target_row)):
            new_w = np.nan_to_num(target_row, nan=0.0)
            turnover[t] = float(np.abs(new_w - w).sum())
            w = new_w

    costs = turnover * cost_bps / 1e4
    net = gross - costs  # le coût est payé à la clôture du jour où l'on trade

    idx = trading_index
    return BacktestResult(
        name=name,
        returns=pd.Series(net, index=idx, name=name),
        gross_returns=pd.Series(gross, index=idx, name=name),
        weights=pd.DataFrame(held, index=idx, columns=prices.columns),
        turnover=pd.Series(turnover, index=idx, name=name),
        costs=pd.Series(costs, index=idx, name=name),
        params={"rebalance": rebalance, "cost_bps": cost_bps},
    )


def common_start(*target_weights: pd.DataFrame) -> pd.Timestamp:
    """Première date où TOUTES les stratégies ont un signal.

    Indispensable pour comparer des stratégies équitablement : une moyenne mobile
    200 jours n'a pas de signal pendant ses 200 premiers jours.
    """
    firsts = [tw.dropna(how="all").index[0] for tw in target_weights]
    return max(firsts)
