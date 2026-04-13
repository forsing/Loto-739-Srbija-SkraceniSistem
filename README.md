# Loto Serbia Portfolio Optimizer

A mathematically honest Streamlit app for **Loto 7/39** that focuses on:

- **portfolio construction**
- **coverage optimization**
- **wheeling systems**
- **Monte Carlo evaluation**
- **fairness diagnostics**
- **responsible play**

## Important

This project does **not** claim to predict future lottery numbers.

If the lottery is fair, each draw is independent.  
The purpose of this app is to improve **portfolio structure** compared with naive overlapping ticket selection, not to create magical forecasting ability.

---

# Features

## 1. Portfolio Generator
Generate multiple tickets using one of these strategies:

- **Coverage-Optimized**
  - maximizes pair/triple coverage
  - reduces overlap between tickets
  - best default strategy

- **Hybrid**
  - mix of coverage-optimized and random tickets

- **Pure Random**
  - baseline / comparison strategy

## 2. Monte Carlo Evaluation
After generating a portfolio, the app estimates:

- probability of at least one **3+** hit
- probability of at least one **4+** hit
- expected total prize
- expected net result

These estimates are based on simulation of the **actual generated portfolio**, not a false independence assumption.

## 3. Empirical Random Baseline
The app compares your portfolio against many random portfolios of the same size and reports percentile-style comparisons such as:

- chance of 3+ vs random
- average best match vs random
- expected prize vs random

## 4. Wheeling System
Build abbreviated wheels with conditional guarantees:

> if enough of your selected key numbers are drawn, at least one ticket is guaranteed to reach a chosen match threshold

Wheeling improves how you organize your chosen numbers.  
It does **not** prove that the chosen numbers are predictive.

## 5. Mathematics Page
Shows:

- exact single-ticket probabilities
- expected value per ticket
- approximate multi-ticket probabilities
- bankroll guidance / responsible play framing

## 6. Fairness Diagnostics
Runs statistical checks on historical draw data to see whether the draw history looks roughly compatible with a fair random process.

These tests are **diagnostic only** and should not be interpreted as proof of exploitability.

## 7. Tracking & History
The app stores generated portfolios and later evaluates them against actual draws.

It tracks:

- best match
- prize value
- historical strategy performance
- empirical comparison to random baselines

---

# Why this project exists

Most lottery apps make misleading claims about:

- hot numbers
- cold numbers
- AI prediction
- trend detection
- hidden patterns

This project takes a different approach:

- admit that lottery prediction is usually not possible
- focus on **combinatorial portfolio quality**
- compare strategies honestly
- show expected value and risk clearly

---

# Core idea

If you buy multiple tickets, the main question is not:

> "Which single ticket is most likely to win?"

Under a fair lottery, all valid tickets are equally likely.

A better question is:

> "How should I structure a set of tickets to reduce redundancy and improve coverage?"

That is where combinatorial optimization and wheeling make sense.

---

# Project Structure

```text
lotto_ai/
├── config.py
├── core/
│   ├── coverage_optimizer.py
│   ├── db.py
│   ├── learner.py
│   ├── math_engine.py
│   ├── models.py
│   ├── tracker.py
│   └── wheeling.py
├── evaluation/
│   ├── backtest.py
│   └── tune_weights.py
├── features/
│   └── features.py
├── gui/
│   └── app.py
└── scraper/
    └── serbia_scraper.py