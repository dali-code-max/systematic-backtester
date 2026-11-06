"""Données de marché : téléchargement (Yahoo Finance), nettoyage et cache local.

Principe : on télécharge une seule fois, on sauvegarde en CSV dans data/,
et les exécutions suivantes relisent le cache (rapide, et reproductible).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Univers multi-actifs par défaut : des ETF liquides avec un long historique.
# Utiliser des ETF (et non des actions choisies aujourd'hui) limite le biais du survivant.
DEFAULT_UNIVERSE: dict[str, str] = {
    "SPY": "Actions US (S&P 500)",
    "EFA": "Actions pays développés hors US",
    "EEM": "Actions émergentes",
    "IEF": "Treasuries US 7-10 ans",
    "TLT": "Treasuries US 20 ans et plus",
    "LQD": "Obligations d'entreprises Investment Grade",
    "GLD": "Or",
    "DBC": "Matières premières",
    "VNQ": "Immobilier coté US",
}

CACHE_DIR = Path("data")


def download_prices(
    tickers: Iterable[str] = DEFAULT_UNIVERSE,
    start: str = "2007-01-01",
    end: str | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Télécharge les prix ajustés (dividendes et splits) et les nettoie.

    Retourne un DataFrame : index = dates, colonnes = tickers.
    """
    tickers = list(tickers)
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"prices_{'-'.join(sorted(tickers))}_{start}_{end or 'latest'}.csv"

    if use_cache and cache_file.exists() and not refresh:
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices[tickers]

    import yfinance as yf  # import local : le reste du package fonctionne sans yfinance

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Aucune donnée reçue de Yahoo Finance : vérifie les tickers et la connexion.")

    # Selon la version de yfinance, les colonnes sont simples ou à deux niveaux (champ, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"Tickers introuvables : {missing}")

    prices = clean_prices(prices[tickers])
    if use_cache:
        prices.to_csv(cache_file)
    return prices


def clean_prices(prices: pd.DataFrame, max_fill_days: int = 5) -> pd.DataFrame:
    """Nettoyage standard d'une matrice de prix.

    - tri chronologique et suppression des dates en double ;
    - suppression des lignes entièrement vides (jours fériés) ;
    - report du dernier prix connu sur au plus `max_fill_days` jours
      (jour férié sur un seul marché), jamais au-delà ;
    - on démarre à la première date où TOUS les actifs ont un prix ;
    - contrôle : aucun prix négatif ou nul.
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")
    prices = prices.ffill(limit=max_fill_days)

    first_common_date = prices.apply(pd.Series.first_valid_index).max()
    prices = prices.loc[first_common_date:]

    if prices.isna().any().any():
        gaps = prices.isna().sum()
        raise ValueError(f"Trous de données trop longs à combler :\n{gaps[gaps > 0]}")
    if (prices <= 0).any().any():
        raise ValueError("Prix négatifs ou nuls détectés : données corrompues.")

    prices.index.name = "date"
    return prices


def synthetic_prices(n_assets: int = 4, n_days: int = 2520, seed: int = 42) -> pd.DataFrame:
    """Prix simulés (marche aléatoire géométrique corrélée).

    Utile pour les tests unitaires et pour travailler hors connexion.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    vols = rng.uniform(0.10, 0.30, n_assets)
    drifts = rng.uniform(0.00, 0.08, n_assets)
    corr = 0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    daily = (drifts - 0.5 * vols**2) / 252 + vols / np.sqrt(252) * shocks
    prices = 100 * np.exp(np.cumsum(daily, axis=0))
    cols = [f"ASSET_{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=pd.Index(dates, name="date"), columns=cols)
