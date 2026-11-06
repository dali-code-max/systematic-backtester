"""Tests du moteur : ils garantissent que les chiffres du backtest sont dignes de confiance.

Lancer avec :  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data import synthetic_prices
from backtester.engine import rebalance_dates, run_backtest
from backtester.metrics import max_drawdown
from backtester.strategies import sma_crossover


@pytest.fixture
def prices():
    return synthetic_prices(n_assets=3, n_days=600, seed=7)


def test_100pct_one_asset_replicates_the_asset(prices):
    """Investi à 100 % dans un actif, sans coûts : on doit retrouver son rendement (dès le 2e jour)."""
    target = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target["ASSET_1"] = 1.0
    res = run_backtest(prices, target, rebalance="D", cost_bps=0)
    asset_ret = prices["ASSET_1"].pct_change()
    np.testing.assert_allclose(res.returns.iloc[1:], asset_ret.iloc[1:], atol=1e-12)
    assert res.returns.iloc[0] == 0  # le premier jour, on achète à la clôture : aucun rendement


def test_no_look_ahead(prices):
    """Un signal daté t n'est détenu qu'à partir de t+1."""
    target = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    switch = prices.index[300]
    target.loc[switch:, "ASSET_2"] = 1.0
    res = run_backtest(prices, target, rebalance="D", cost_bps=0)
    assert res.weights.loc[switch, "ASSET_2"] == 0.0
    assert res.weights.iloc[301]["ASSET_2"] == pytest.approx(1.0)


def test_perfect_foresight_signal_is_neutralised(prices):
    """Un signal qui « connaît » le rendement du jour même ne doit pas devenir une machine à cash.

    On construit un signal tricheur : long l'actif les jours où il monte.
    Si le moteur oubliait le décalage, ce backtest afficherait un Sharpe astronomique.
    """
    daily = prices.pct_change()
    cheat = (daily > 0).astype(float) / prices.shape[1]
    res = run_backtest(prices, cheat, rebalance="D", cost_bps=0)
    sharpe = res.returns.mean() / res.returns.std() * np.sqrt(252)
    assert sharpe < 3


def test_transaction_costs(prices):
    """Passer de 100 % cash à 100 % investi coûte exactement cost_bps sur le montant échangé."""
    target = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target["ASSET_1"] = 1.0
    res = run_backtest(prices, target, rebalance="never", cost_bps=10)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.costs.iloc[0] == pytest.approx(0.001)
    assert res.turnover.iloc[1:].sum() == 0


def test_buy_and_hold_weights_drift(prices):
    """Sans rebalancement, la valeur finale = moyenne des performances des actifs."""
    target = pd.DataFrame(1 / 3, index=prices.index, columns=prices.columns)
    res = run_backtest(prices, target, rebalance="never", cost_bps=0)
    expected = (prices.iloc[-1] / prices.iloc[0]).mean()
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-10)
    # Les poids ont dérivé : ils ne sont plus égaux à la fin
    assert res.weights.iloc[-1].std() > 0


def test_monthly_rebalance_dates(prices):
    dates = rebalance_dates(prices.index, "M")
    months = prices.index.to_period("M").unique()
    assert len(dates) == len(months)
    assert all(d == prices.index[prices.index.to_period("M") == d.to_period("M")][-1] for d in dates)


def test_sma_has_no_signal_during_warmup(prices):
    w = sma_crossover(prices, fast=20, slow=100)
    assert w.iloc[:99].isna().all().all()
    assert w.iloc[99:].notna().all().all()


def test_max_drawdown_simple_case():
    r = pd.Series([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.50)
