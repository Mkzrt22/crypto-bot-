from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    allowed: bool
    reason: str
    adjusted_fraction: float = 1.0


class RiskManager:
    def __init__(self, config):
        self.config = config

    def check_buy(self, portfolio, symbol: str, price: float, fraction: float) -> RiskCheck:
        # Max drawdown kill-switch
        if portfolio.get_drawdown() <= -self.config.MAX_DRAWDOWN_PCT:
            return RiskCheck(False, "Max drawdown reached — trading halted")

        total = portfolio.get_total_value()
        pos = portfolio.get_position(symbol)
        current_pos_val = pos.amount * price
        proposed_spend = portfolio.balance * fraction
        new_pos_pct = (current_pos_val + proposed_spend) / max(total, 1.0)

        if new_pos_pct > self.config.MAX_POSITION_PCT:
            max_spend = total * self.config.MAX_POSITION_PCT - current_pos_val
            if max_spend <= 0:
                return RiskCheck(False, f"Position limit {self.config.MAX_POSITION_PCT*100:.0f}% reached")
            adj = max_spend / max(portfolio.balance, 1.0)
            return RiskCheck(True, "Position size adjusted", adjusted_fraction=min(adj, fraction))

        if portfolio.balance < 10:
            return RiskCheck(False, "Insufficient balance")

        return RiskCheck(True, "OK", adjusted_fraction=fraction)

    def check_sell(self, portfolio, symbol: str) -> RiskCheck:
        pos = portfolio.get_position(symbol)
        if pos.amount <= 0:
            return RiskCheck(False, "No position to sell")
        return RiskCheck(True, "OK")

    def check_stop_loss(self, portfolio, symbol: str, price: float) -> bool:
        pos = portfolio.get_position(symbol)
        if pos.amount <= 0 or pos.entry_price <= 0:
            return False
        return (price - pos.entry_price) / pos.entry_price <= -self.config.STOP_LOSS_PCT

    def check_take_profit(self, portfolio, symbol: str, price: float) -> bool:
        pos = portfolio.get_position(symbol)
        if pos.amount <= 0 or pos.entry_price <= 0:
            return False
        return (price - pos.entry_price) / pos.entry_price >= self.config.TAKE_PROFIT_PCT
