# ES Negative Volatility Targeting

This folder contains an initial strategy project focused on volatility targeting in the US equity market through ES futures.

The reason for starting here is that equity exposure is the natural benchmark for most investors. Whether explicitly or implicitly, many portfolios are judged against the S&P 500 or, more broadly, global equity markets. In the futures space, ES is one of the cleanest and most liquid ways to represent that benchmark, which makes it a strong starting point for systematic research.

The project is also motivated by a simple theoretical view: if markets are broadly efficient, persistent excess returns are hard to obtain, which makes disciplined management of market exposure especially relevant. From a CAPM perspective, equity exposure is a natural benchmark, and futures provide an efficient way to scale that exposure through leverage while keeping implementation flexible.

It is therefore a useful market for studying risk-managed exposure. Volatility regimes, drawdowns, and trend persistence can materially affect outcomes, so the question is not only whether equities rise over time, but whether exposure to the equity market can be managed more efficiently.

For the moment, this folder contains two working files:

- `NegVolDirection - Python Backtest.ipynb`  
  Research notebook used to study the strategy logic and evaluate its historical behavior.

- `VIXDirection - Python Backtest.ipynb`  
  Research notebook used to study the VIX-based volatility-targeting variant and compare its behavior with the downside-volatility approach.
- `NegVolDirection - MT5 Deployment.ipynb`  
  Notebook used to connect with MetaTrader 5 and monitor the live account implementation.

This project is still at an early stage. The main goal for now is to keep a clean link between research, deployment, and live monitoring.