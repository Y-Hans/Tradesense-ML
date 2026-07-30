"""Behaviour generator and cognitive bias injector for synthetic trading scenarios."""

import random
from typing import Any

from tradesense_ml.domain.schemas.synthetic import BiasInjectionConfig, BiasType
from tradesense_ml.domain.schemas.trade import Side, Trade, TradeExecution, TradeOrder
from tradesense_ml.pipelines.generation.base import BaseBiasInjector


class BehaviourGenerator(BaseBiasInjector):
    """Behavioral bias injector simulating cognitive trading biases and psychology."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Initialize behaviour generator with optional random number generator instance."""
        self._rng = rng or random.Random(42)

    def inject_bias(self, trade: Trade, bias_config: BiasInjectionConfig) -> Trade:
        """Modify trade parameters to reflect specific cognitive behavioral bias."""
        bias_type = bias_config.bias_type
        intensity = bias_config.intensity
        rng = self._rng

        # Copy mutable attributes
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        quantity = trade.quantity
        sl = trade.initial_stop_loss
        tp = trade.initial_take_profit
        pnl = trade.pnl
        pnl_pct = trade.pnl_percentage
        orders = list(trade.orders)
        tags = list(trade.tags)
        metadata: dict[str, Any] = dict(trade.metadata)

        tags.append(f"bias:{bias_type.value.lower()}")
        metadata["bias_applied"] = bias_type.value
        metadata["bias_intensity"] = intensity
        metadata["bias_description"] = bias_config.description

        is_long = trade.side in (Side.LONG, Side.BUY)

        if bias_type == BiasType.DISCIPLINE or bias_type == BiasType.NO_BIAS_CLEAN:
            # Disciplined trade: clean risk management
            metadata["plan_followed"] = True
            metadata["risk_discipline_score"] = round(rng.uniform(8.5, 10.0), 2)

        elif bias_type in (BiasType.FOMO, BiasType.CHASING_PRICE):
            # Chased entry: worse price by intensity %
            price_shift = entry_price * (0.005 + 0.02 * intensity)
            entry_price = round(
                entry_price + price_shift if is_long else entry_price - price_shift, 2
            )
            tags.append("chased_entry")
            metadata["plan_followed"] = False
            metadata["chased_ticks"] = round(price_shift, 2)

        elif bias_type == BiasType.REVENGE_TRADING:
            # Oversized position after loss
            quantity = round(quantity * (1.5 + 2.5 * intensity), 2)
            tags.append("oversized_position")
            metadata["plan_followed"] = False
            metadata["revenge_multiplier"] = round(1.5 + 2.5 * intensity, 2)

        elif bias_type == BiasType.OVERCONFIDENCE:
            # High leverage / huge size and tight/ignored stop loss
            quantity = round(quantity * (2.0 + 3.0 * intensity), 2)
            if sl is not None:
                sl = round(
                    (
                        sl - (entry_price * 0.03 * intensity)
                        if is_long
                        else sl + (entry_price * 0.03 * intensity)
                    ),
                    2,
                )
            metadata["plan_followed"] = False
            tags.append("overconfident")

        elif bias_type in (BiasType.FEAR, BiasType.HESITATION):
            # Reduced position size, premature exit
            quantity = max(1.0, round(quantity * (0.2 + 0.3 * (1.0 - intensity)), 2))
            if exit_price is not None:
                # Cut early
                exit_price = round(entry_price + (exit_price - entry_price) * 0.3, 2)
            tags.append("premature_exit")
            metadata["plan_followed"] = False

        elif bias_type == BiasType.POSITION_SIZING_ERRORS:
            # Sizing error: erratic quantity
            multiplier = rng.choice([0.1, 5.0, 10.0])
            quantity = max(1.0, round(quantity * multiplier, 2))
            tags.append("position_size_error")
            metadata["plan_followed"] = False

        elif bias_type == BiasType.MOVING_STOP_LOSS:
            # Widen stop loss mid-trade
            if sl is not None:
                sl = round(
                    (
                        sl - (entry_price * 0.04 * intensity)
                        if is_long
                        else sl + (entry_price * 0.04 * intensity)
                    ),
                    2,
                )
            tags.append("moved_stop_loss")
            metadata["plan_followed"] = False

        elif bias_type == BiasType.IGNORING_PLAN:
            # Clear stop loss and take profit
            sl = None
            tp = None
            tags.append("no_stop_loss")
            metadata["plan_followed"] = False

        elif bias_type == BiasType.IMPULSIVENESS:
            # Arbitrary stop/tp levels
            sl = round(
                (
                    entry_price * (0.97 - 0.02 * intensity)
                    if is_long
                    else entry_price * (1.03 + 0.02 * intensity)
                ),
                2,
            )
            tp = round(
                (
                    entry_price * (1.01 + 0.01 * intensity)
                    if is_long
                    else entry_price * (0.99 - 0.01 * intensity)
                ),
                2,
            )
            metadata["plan_followed"] = False

        # Scale execution fills consistently with new quantity and prices
        updated_executions = []
        scale_ratio = quantity / trade.quantity if trade.quantity > 0 else 1.0

        if trade.executions:
            # First execution is entry fill
            e_in = trade.executions[0]
            updated_executions.append(
                TradeExecution(
                    execution_id=e_in.execution_id,
                    price=entry_price,
                    quantity=quantity,
                    timestamp=e_in.timestamp,
                    fee=round(e_in.fee * scale_ratio, 2),
                )
            )

            # Remaining executions are exit fills
            exit_execs = trade.executions[1:]
            if exit_execs:
                remaining_qty = quantity
                for idx, e_out in enumerate(exit_execs):
                    if idx == len(exit_execs) - 1:
                        fill_qty = round(remaining_qty, 2)
                    else:
                        fill_qty = round(e_out.quantity * scale_ratio, 2)
                        remaining_qty -= fill_qty

                    updated_executions.append(
                        TradeExecution(
                            execution_id=e_out.execution_id,
                            price=exit_price if exit_price is not None else e_out.price,
                            quantity=fill_qty,
                            timestamp=e_out.timestamp,
                            fee=round(e_out.fee * scale_ratio, 2),
                        )
                    )

        # Recalculate PnL if exit_price is present
        if exit_price is not None:
            total_fees = sum(ex.fee for ex in updated_executions) if updated_executions else 0.0
            raw_pnl = (
                (exit_price - entry_price) * quantity
                if is_long
                else (entry_price - exit_price) * quantity
            )
            pnl = round(raw_pnl - total_fees, 2)
            denom = entry_price * quantity
            pnl_pct = round((pnl / denom) * 100.0, 2) if denom > 0 else 0.0

        # Update orders list with current sl/tp
        updated_orders = [
            TradeOrder(
                order_id=o.order_id,
                side=o.side,
                limit_price=o.limit_price,
                stop_price=o.stop_price,
                stop_loss=sl,
                take_profit=tp,
            )
            for o in orders
        ]

        return Trade(
            trade_id=trade.trade_id,
            user_id=trade.user_id,
            symbol=trade.symbol,
            side=trade.side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            initial_stop_loss=sl,
            initial_take_profit=tp,
            pnl=pnl,
            pnl_percentage=pnl_pct,
            timeframe=trade.timeframe,
            entry_timestamp=trade.entry_timestamp,
            exit_timestamp=trade.exit_timestamp,
            orders=updated_orders,
            executions=updated_executions,
            tags=tags,
            metadata=metadata,
        )
