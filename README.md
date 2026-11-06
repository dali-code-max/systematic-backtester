# Systematic Backtester

Un framework de backtesting multi-actifs en Python, construit autour d'une idée simple :
**la plupart des backtests sont trop beaux pour être vrais.** Ce projet cherche à mesurer
honnêtement ce qu'une stratégie aurait rapporté, coûts compris, et à quel point ce
résultat est robuste.

> 🚧 Projet en construction — étape 1 sur 5.

## Principes de conception

- **Une stratégie ne produit que des poids cibles.** Le moteur gère tout le reste.
- **Pas de look-ahead bias** : un signal calculé à la clôture de `t` n'est détenu qu'à partir de `t+1`. Un test unitaire le vérifie.
- **Les poids dérivent** entre deux rebalancements, comme dans un vrai portefeuille.
- **Coûts de transaction** appliqués à chaque montant échangé.
- **Données publiques uniquement** : ETF multi-actifs via Yahoo Finance.

## Structure

```
backtester/
  data.py        téléchargement, nettoyage et cache des prix
  strategies.py  stratégies (prix -> poids cibles)
  engine.py      exécution : décalage, dérive, rebalancement, coûts
  metrics.py     CAGR, volatilité, Sharpe, Sortino, drawdown, Calmar, turnover
  plotting.py    graphiques de performance et de drawdown
tests/           tests unitaires du moteur
run_step1.py     premier backtest de bout en bout
```

## Démarrage rapide

```bash
pip install -r requirements.txt
python run_step1.py            # données réelles
python run_step1.py --offline  # données simulées
pytest -q                      # tests
```

## Feuille de route

- [x] Étape 1 — données, moteur, premières stratégies
- [ ] Étape 2 — métriques complètes et rapports
- [ ] Étape 3 — momentum, risk parity, volatility targeting…
- [ ] Étape 4 — robustesse : walk-forward, sensibilité, bootstrap, Deflated Sharpe Ratio
- [ ] Étape 5 — interface Streamlit

## Avertissement

Projet personnel à but pédagogique. Rien ici ne constitue un conseil en investissement.
