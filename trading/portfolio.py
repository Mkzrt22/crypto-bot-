from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    amount: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0

    @property
    def value(self) -> float:
        return self.amount * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def pnl_usd(self) -> float:
        return self.amount * (self.current_price - self.entry_price)


class Portfolio:
    def __init__(self, initial_balance: float, config):
        self.initial_balance = initial_balance
        self.balance = float(initial_balance)
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trade_log: list = []
        self.peak_value = float(initial_balance)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_total_value(self) -> float:
        return self.balance + sum(p.value for p in self.positions.values())

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def get_roi(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.get_total_value() - self.initial_balance) / self.initial_balance

    def get_drawdown(self) -> float:
        pv = self.get_total_value()
        if self.peak_value <= 0:
            return 0.0
        return (pv - self.peak_value) / self.peak_value

    def update_price(self, symbol: str, price: float):
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    # ------------------------------------------------------------------
    # Execution (paper trading)
    # ------------------------------------------------------------------

    def buy(self, symbol: str, price: float, amount: float, fee: float = 0.001) -> dict:
        cost = amount * price
        fee_amt = cost * fee
        total = cost + fee_amt

        if total > self.balance:
            amount = self.balance * (1 - fee) / max(price, 1e-8)
            cost = amount * price
            fee_amt = cost * fee
            total = cost + fee_amt

        if amount <= 0:
            return {}

        self.balance -= total
        pos = self.get_position(symbol)
        if pos.amount > 0:
            new_total = pos.amount + amount
            pos.entry_price = (pos.entry_price * pos.amount + price * amount) / new_total
        else:
            pos.entry_price = price
        pos.amount += amount
        pos.current_price = price

        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol, "side": "buy",
            "price": price, "amount": amount,
            "value": cost, "fee": fee_amt,
        }
        self.trade_log.append(trade)
        self.peak_value = max(self.peak_value, self.get_total_value())
        return trade

    def sell(self, symbol: str, price: float, fraction: float = 1.0, fee: float = 0.001) -> dict:
        pos = self.get_position(symbol)
        if pos.amount <= 0:
            return {}

        sell_amt = pos.amount * fraction
        gross = sell_amt * price
        fee_amt = gross * fee
        net = gross - fee_amt
        pnl = sell_amt * (price - pos.entry_price)

        self.balance += net
        pos.amount -= sell_amt
        pos.current_price = price
        if pos.amount < 1e-8:
            pos.amount = 0.0
            pos.entry_price = 0.0

        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol, "side": "sell",
            "price": price, "amount": sell_amt,
            "value": gross, "fee": fee_amt, "pnl": pnl,
        }
        self.trade_log.append(trade)
        return trade

    def get_snapshot(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "balance": self.balance,
            "total_value": self.get_total_value(),
            "roi": self.get_roi(),
            "drawdown": self.get_drawdown(),
            "positions": {
                sym: {
                    "amount": p.amount,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "value": p.value,
                    "pnl_pct": p.pnl_pct,
                    "pnl_usd": p.pnl_usd,
                }
                for sym, p in self.positions.items()
                if p.amount > 0
            },
        }
