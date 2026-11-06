"""Mesures de performance et de risque, à partir de rendements quotidiens."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    """Taux de croissance annuel composé."""
    growth = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else np.nan


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Sharpe annualisé ; risk_free est un taux annuel."""
    excess = returns - risk_free / TRADING_DAYS
    vol = excess.std(ddof=1)
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Comme le Sharpe, mais ne pénalise que la volatilité à la baisse."""
    excess = returns - risk_free / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan


def drawdown(returns: pd.Series) -> pd.Series:
    """Perte depuis le plus haut historique, à chaque date (valeurs <= 0)."""
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(returns)
    return cagr(returns) / abs(mdd) if mdd < 0 else np.nan


def annual_turnover(turnover: pd.Series) -> float:
    """Montant échangé par an, en multiple du portefeuille (1.0 = 100 %).

    Convention : somme des |variations de poids| (achats + ventes).
    """
    years = len(turnover) / TRADING_DAYS
    return float(turnover.sum() / years) if years > 0 else np.nan


def summary(result) -> dict:
    """Tableau de bord d'une stratégie (objet BacktestResult)."""
    r = result.returns
    return {
        "CAGR": cagr(r),
        "Volatilité": annual_volatility(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Max drawdown": max_drawdown(r),
        "Calmar": calmar_ratio(r),
        "Turnover annuel": annual_turnover(result.turnover),
        "Coûts annuels": float(result.costs.sum() / (len(r) / TRADING_DAYS)),
        "Expo. moyenne": float(result.weights.sum(axis=1).mean()),
    }


def summary_table(results) -> pd.DataFrame:
    """Une ligne par stratégie."""
    return pd.DataFrame({res.name: summary(res) for res in results}).T


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Version lisible pour l'affichage console."""
    pct = ["CAGR", "Volatilité", "Max drawdown", "Turnover annuel", "Coûts annuels", "Expo. moyenne"]
    out = table.copy().astype(object)
    for col in table.columns:
        if col in pct:
            out[col] = table[col].map(lambda x: f"{x:.1%}")
        else:
            out[col] = table[col].map(lambda x: f"{x:.2f}")
    return out
