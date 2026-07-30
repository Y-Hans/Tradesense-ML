"""Trade execution simulator generating realistic, mathematically valid trades."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from tradesense_ml.domain.schemas.market_context import MarketContext, MarketRegime
from tradesense_ml.domain.schemas.trade import Side, TimeFrame, Trade, TradeExecution, TradeOrder
from tradesense_ml.pipelines.generation.base import BaseTradeSimulator
from tradesense_ml.pipelines.generation.outcome_generator import TradeOutcome, TradeOutcomeGenerator


class TradeSimulator(BaseTradeSimulator):
    """Simulator for generating realistic, mathematically consistent trade executions."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Initialize trade simulator with optional random number generator instance."""
        self._rng = rng or random.Random(42)
        self.outcome_generator = TradeOutcomeGenerator(rng=self._rng)

    def simulate_trade(self, context: MarketContext, user_id: str = "trader_001") -> Trade:
        """Simulate realistic trade entry/exit against market context."""
        return self.simulate_trade_with_outcome(context, user_id=user_id)

    def simulate_trade_with_outcome(
        self,
        context: MarketContext,
        user_id: str = "trader_001",
        outcome: TradeOutcome | None = None,
        risk_profile: str = "moderate",
        trade_id: str | None = None,
    ) -> Trade:
        """Simulate trade with explicit outcome, risk profile, and trade ID."""
        rng = self._rng
        symbol = context.symbol
        regime = context.regime
        tid = trade_id or f"trd_{rng.randint(100000, 999999)}"

        base_price = context.context_metadata.get("base_price", 100.0)
        atr = context.indicators.atr_14 or (base_price * 0.02)

        # Decide side based on regime trend
        if regime in (
            MarketRegime.BULLISH_TREND,
            MarketRegime.HIGH_VOLATILITY_BREAKOUT,
            MarketRegime.RECOVERY,
        ):
            side = Side.LONG if rng.random() < 0.75 else Side.SHORT
        elif regime in (
            MarketRegime.BEARISH_TREND,
            MarketRegime.BEARISH_BREAKDOWN,
            MarketRegime.FLASH_CRASH,
        ):
            side = Side.SHORT if rng.random() < 0.75 else Side.LONG
        else:
            side = Side.LONG if rng.random() < 0.5 else Side.SHORT

        is_long = side in (Side.LONG, Side.BUY)

        # Base entry price with slight noise
        entry_price = round(base_price * (1.0 + rng.uniform(-0.005, 0.005)), 2)

        # Position sizing (quantity) based on risk profile
        base_size = 100.0 if base_price < 200 else 10.0
        if risk_profile == "conservative":
            quantity = round(base_size * rng.uniform(0.5, 1.0), 2)
        elif risk_profile == "aggressive":
            quantity = round(base_size * rng.uniform(2.0, 5.0), 2)
        else:  # moderate
            quantity = round(base_size * rng.uniform(1.0, 2.0), 2)

        # Calculate initial stop loss and take profit
        sl_distance = round(atr * rng.uniform(1.2, 2.0), 2)
        tp_distance = round(sl_distance * rng.uniform(1.5, 3.0), 2)

        initial_sl = round(entry_price - sl_distance if is_long else entry_price + sl_distance, 2)
        initial_tp = round(entry_price + tp_distance if is_long else entry_price - tp_distance, 2)

        # Determine outcome if not provided
        if outcome is None:
            outcome = self.outcome_generator.determine_outcome(
                market_context=context,
                bias_type=context.context_metadata.get("bias_applied", "NO_BIAS_CLEAN"),
                risk_profile=risk_profile,
            )

        # Exit price calculation based on outcome
        if outcome in (
            TradeOutcome.DISCIPLINED_WINNER,
            TradeOutcome.LUCKY_WINNER,
            TradeOutcome.RECKLESS_WINNER,
        ):
            gain = tp_distance * rng.uniform(0.8, 1.2)
            exit_price = round(entry_price + gain if is_long else entry_price - gain, 2)
        elif outcome in (TradeOutcome.PREMATURE_EXIT, TradeOutcome.MISSED_OPPORTUNITY):
            gain = tp_distance * rng.uniform(0.1, 0.3)
            exit_price = round(entry_price + gain if is_long else entry_price - gain, 2)
        elif outcome in (
            TradeOutcome.DISCIPLINED_LOSER,
            TradeOutcome.FOMO_LOSS,
            TradeOutcome.REVENGE_LOSS,
        ):
            loss = sl_distance * rng.uniform(0.9, 1.1)
            exit_price = round(entry_price - loss if is_long else entry_price + loss, 2)
        else:  # TradeOutcome.OVERLEVERAGED_LIQUIDATION
            loss = sl_distance * rng.uniform(2.0, 3.5)
            exit_price = round(entry_price - loss if is_long else entry_price + loss, 2)

        # Executions (fills, partial exits, fees, slippage)
        base_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        entry_time = base_dt + timedelta(minutes=rng.randint(0, 100_000))
        duration_minutes = rng.randint(5, 240)
        exit_time = entry_time + timedelta(minutes=duration_minutes)

        fee_rate = 0.0005  # 0.05%
        entry_fee = round(entry_price * quantity * fee_rate, 2)

        # Generate 1 to 3 exit fills for partial exit modeling
        num_fills = rng.choice([1, 1, 2, 3])
        executions: list[TradeExecution] = []

        # Entry execution
        executions.append(
            TradeExecution(
                execution_id=f"exec_in_{tid}",
                price=entry_price,
                quantity=quantity,
                timestamp=entry_time,
                fee=entry_fee,
            )
        )

        # Exit executions (partial fills summing to quantity)
        remaining_qty = quantity
        for i in range(num_fills):
            if i == num_fills - 1:
                fill_qty = round(remaining_qty, 2)
            else:
                fill_qty = round(quantity / num_fills, 2)
                remaining_qty -= fill_qty

            fill_price = round(exit_price + rng.uniform(-0.02, 0.02), 2)
            fill_time = entry_time + timedelta(minutes=int(duration_minutes * (i + 1) / num_fills))
            fill_fee = round(fill_price * fill_qty * fee_rate, 2)

            executions.append(
                TradeExecution(
                    execution_id=f"exec_out_{tid}_{i+1}",
                    price=fill_price,
                    quantity=fill_qty,
                    timestamp=fill_time,
                    fee=fill_fee,
                )
            )

        # Weighted average exit price from fills
        exit_fills = executions[1:]
        total_exit_qty = sum(e.quantity for e in exit_fills)
        weighted_exit_price = round(
            sum(e.price * e.quantity for e in exit_fills) / total_exit_qty, 2
        )
        total_fees = round(sum(e.fee for e in executions), 2)

        # Calculate exact PnL and PnL %
        raw_pnl = (
            (weighted_exit_price - entry_price) * quantity
            if is_long
            else (entry_price - weighted_exit_price) * quantity
        )
        pnl = round(raw_pnl - total_fees, 2)
        invested_capital = entry_price * quantity
        pnl_pct = round((pnl / invested_capital) * 100.0, 2)

        # Risk %, Reward %, R multiple calculation
        risk_amount = abs(entry_price - initial_sl) * quantity
        reward_amount = abs(initial_tp - entry_price) * quantity
        risk_pct = (
            round((risk_amount / invested_capital) * 100.0, 2) if invested_capital > 0 else 0.0
        )
        reward_pct = (
            round((reward_amount / invested_capital) * 100.0, 2) if invested_capital > 0 else 0.0
        )
        r_multiple = round(pnl / risk_amount, 2) if risk_amount > 0 else 0.0

        orders = [
            TradeOrder(
                order_id=f"ord_entry_{tid}",
                side=side,
                limit_price=entry_price,
                stop_loss=initial_sl,
                take_profit=initial_tp,
            )
        ]

        metadata: dict[str, Any] = {
            "outcome": outcome.value,
            "risk_profile": risk_profile,
            "risk_pct": risk_pct,
            "reward_pct": reward_pct,
            "r_multiple": r_multiple,
            "holding_duration_minutes": duration_minutes,
            "total_fees": total_fees,
            "is_win": pnl > 0,
            "is_breakeven": abs(pnl) < 1.0,
            "partial_exits_count": num_fills,
        }

        tags = [
            f"side:{side.value.lower()}",
            f"outcome:{outcome.value}",
            f"risk:{risk_profile}",
        ]
        if pnl > 0:
            tags.append("winner")
        elif pnl < 0:
            tags.append("loser")

        return Trade(
            trade_id=tid,
            user_id=user_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=weighted_exit_price,
            quantity=quantity,
            initial_stop_loss=initial_sl,
            initial_take_profit=initial_tp,
            pnl=pnl,
            pnl_percentage=pnl_pct,
            timeframe=TimeFrame.M5,
            entry_timestamp=entry_time,
            exit_timestamp=exit_time,
            orders=orders,
            executions=executions,
            tags=tags,
            metadata=metadata,
        )
