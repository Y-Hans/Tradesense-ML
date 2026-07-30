"""Unit tests for TradeSimulator handling Long/Short, fills, fees, slippage, and math validity."""

import random

import pytest

from tradesense_ml.domain.schemas.market_context import MarketRegime
from tradesense_ml.domain.schemas.synthetic import MarketScenarioConfig
from tradesense_ml.domain.schemas.trade import Side
from tradesense_ml.pipelines.generation.market_generator import MarketGenerator
from tradesense_ml.pipelines.generation.trade_simulator import TradeSimulator


def test_trade_simulator_long_and_short_math() -> None:
    """Test TradeSimulator produces mathematically consistent Long and Short trades."""
    rng = random.Random(42)
    market_gen = MarketGenerator(rng=rng)
    trade_sim = TradeSimulator(rng=rng)

    ctx = market_gen.generate_market(
        MarketScenarioConfig(
            scenario_id="s1", symbol="NVDA", regime=MarketRegime.BULLISH_TREND, seed=42
        )
    )

    trade = trade_sim.simulate_trade(ctx, user_id="u123")

    assert trade.symbol == "NVDA"
    assert trade.entry_price > 0
    assert trade.quantity > 0
    assert trade.exit_price is not None and trade.exit_price > 0
    assert trade.pnl is not None
    assert trade.pnl_percentage is not None
    assert len(trade.executions) >= 2  # at least entry + 1 exit fill

    # Validate fill quantity sum
    exit_fills = [e for e in trade.executions if e.execution_id.startswith("exec_out")]
    total_exit_qty = sum(e.quantity for e in exit_fills)
    assert pytest.approx(total_exit_qty, abs=1e-3) == trade.quantity

    # Validate mathematical PnL equation
    is_long = trade.side in (Side.LONG, Side.BUY)
    raw_pnl = (
        (trade.exit_price - trade.entry_price) * trade.quantity
        if is_long
        else (trade.entry_price - trade.exit_price) * trade.quantity
    )
    total_fees = sum(e.fee for e in trade.executions)
    expected_pnl = round(raw_pnl - total_fees, 2)
    assert pytest.approx(trade.pnl, abs=0.1) == expected_pnl
