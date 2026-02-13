"""Portfolio Manager — tracks positions, executes trades, computes PnL."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import INITIAL_CASH, TRADE_FRACTION


@dataclass
class _Position:
    """Internal record for a single ticker position."""
    shares: float = 0.0
    avg_cost: float = 0.0


class Portfolio:
    """Virtual portfolio supporting multi-ticker Buy/Sell/Hold execution.

    Each agent (RL, LLM) maintains its own ``Portfolio`` instance so that
    their equity curves are fully independent.

    Parameters
    ----------
    initial_cash : float
        Starting cash balance (default from config).
    trade_fraction : float
        Fraction of available cash (buy) or holdings (sell) per trade.
    """

    def __init__(
        self,
        initial_cash: float = INITIAL_CASH,
        trade_fraction: float = TRADE_FRACTION,
    ) -> None:
        self.cash: float = initial_cash
        self._initial_cash = initial_cash
        self._trade_fraction = trade_fraction
        self._positions: dict[str, _Position] = {}

    # ── Trade execution ──────────────────────────────────────────────────────

    def execute_trade(self, ticker: str, action: str, price: float) -> dict:
        """Execute a trade and return a receipt dict.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. ``"AAPL"``.
        action : str
            One of ``"BUY"``, ``"SELL"``, ``"HOLD"`` (case-insensitive).
        price : float
            Current market price for the ticker.

        Returns
        -------
        dict
            ``{"ticker", "action", "price", "shares_traded", "cost"}``
        """
        action = action.upper()
        pos = self._positions.setdefault(ticker, _Position())
        shares_traded = 0.0
        cost = 0.0

        if action == "BUY" and price > 0:
            invest = self.cash * self._trade_fraction
            if invest > 0:
                bought = invest / price
                # Update weighted average cost
                total_shares = pos.shares + bought
                if total_shares > 0:
                    pos.avg_cost = (
                        (pos.avg_cost * pos.shares + invest) / total_shares
                    )
                pos.shares = total_shares
                self.cash -= invest
                shares_traded = bought
                cost = invest

        elif action == "SELL" and pos.shares > 0:
            sell_shares = pos.shares * self._trade_fraction
            proceeds = sell_shares * price
            pos.shares -= sell_shares
            self.cash += proceeds
            shares_traded = -sell_shares
            cost = -proceeds

        return {
            "ticker": ticker,
            "action": action,
            "price": price,
            "shares_traded": shares_traded,
            "cost": cost,
        }

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_equity(self, prices: dict[str, float]) -> float:
        """Total portfolio value: cash + sum of (shares × price) per ticker."""
        holdings_value = sum(
            pos.shares * prices.get(ticker, 0.0)
            for ticker, pos in self._positions.items()
        )
        return self.cash + holdings_value

    def get_holdings_value(self, prices: dict[str, float]) -> float:
        """Total value of held positions (excluding cash)."""
        return sum(
            pos.shares * prices.get(ticker, 0.0)
            for ticker, pos in self._positions.items()
        )

    def get_unrealised_pnl(self, prices: dict[str, float]) -> float:
        """Total unrealised PnL across all positions."""
        return sum(
            pos.shares * (prices.get(ticker, 0.0) - pos.avg_cost)
            for ticker, pos in self._positions.items()
            if pos.shares > 0
        )

    def get_position(self, ticker: str) -> dict:
        """Return position info for a single ticker."""
        pos = self._positions.get(ticker, _Position())
        return {"shares": pos.shares, "avg_cost": pos.avg_cost}

    def snapshot(self, prices: dict[str, float]) -> dict:
        """Full portfolio state for logging."""
        return {
            "cash": self.cash,
            "equity": self.get_equity(prices),
            "holdings_value": self.get_holdings_value(prices),
            "unrealised_pnl": self.get_unrealised_pnl(prices),
            "positions": {
                tk: {"shares": p.shares, "avg_cost": p.avg_cost}
                for tk, p in self._positions.items()
                if p.shares > 0
            },
        }
