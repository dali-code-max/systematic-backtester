"""Étape 1 : premier backtest de bout en bout.

Utilisation (depuis la racine du projet) :
    python run_step1.py              # données réelles Yahoo Finance
    python run_step1.py --offline    # données simulées, sans connexion
    python run_step1.py --cost-bps 10 --rebalance W
"""

import argparse
from pathlib import Path

from backtester.data import DEFAULT_UNIVERSE, download_prices, synthetic_prices
from backtester.engine import common_start, run_backtest
from backtester.metrics import format_table, summary_table
from backtester.plotting import plot_performance
from backtester.strategies import equal_weight, sma_crossover


def main():
    parser = argparse.ArgumentParser(description="Étape 1 du Systematic Backtester")
    parser.add_argument("--offline", action="store_true", help="utiliser des prix simulés")
    parser.add_argument("--start", default="2007-01-01", help="début de l'historique téléchargé")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="coût par transaction (bps)")
    parser.add_argument("--rebalance", default="M", choices=["D", "W", "M", "Q"])
    args = parser.parse_args()

    # 1) Données
    if args.offline:
        prices = synthetic_prices(n_assets=6, n_days=252 * 15)
    else:
        prices = download_prices(DEFAULT_UNIVERSE, start=args.start)
    print(f"Données : {prices.shape[1]} actifs, du {prices.index[0]:%d/%m/%Y} au {prices.index[-1]:%d/%m/%Y}")

    # 2) Signaux (poids cibles) : chaque stratégie ne fait que ça
    w_equal = equal_weight(prices)
    w_sma = sma_crossover(prices, fast=50, slow=200)

    # 3) Même date de départ pour tout le monde (la SMA 200 a besoin de 200 jours d'historique)
    start = common_start(w_equal, w_sma)
    print(f"Début commun des backtests : {start:%d/%m/%Y}\n")

    # 4) Exécution par le moteur
    results = [
        run_backtest(prices, w_equal, rebalance="never", cost_bps=args.cost_bps,
                     start=start, name="Buy & hold 1/N"),
        run_backtest(prices, w_equal, rebalance=args.rebalance, cost_bps=args.cost_bps,
                     start=start, name="1/N rebalancé"),
        run_backtest(prices, w_sma, rebalance=args.rebalance, cost_bps=args.cost_bps,
                     start=start, name="SMA 50/200"),
    ]

    # 5) Résultats
    table = summary_table(results)
    print(format_table(table).to_string())

    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/step1_metrics.csv")
    path = plot_performance(results, "reports/step1_performance.png",
                            title="Étape 1 : buy & hold, 1/N rebalancé et SMA 50/200")
    print(f"\nTableau enregistré : reports/step1_metrics.csv")
    print(f"Graphique enregistré : {path}")


if __name__ == "__main__":
    main()
