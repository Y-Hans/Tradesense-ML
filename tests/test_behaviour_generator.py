"""Unit tests for BehaviourGenerator and bias injection across all 11 cognitive behaviors."""

import random

from tradesense_ml.domain.schemas.market_context import MarketRegime
from tradesense_ml.domain.schemas.synthetic import (
    BiasInjectionConfig,
    BiasType,
    MarketScenarioConfig,
)
from tradesense_ml.pipelines.generation.behaviour_generator import BehaviourGenerator
from tradesense_ml.pipelines.generation.market_generator import MarketGenerator
from tradesense_ml.pipelines.generation.trade_simulator import TradeSimulator


def test_behaviour_generator_all_biases() -> None:
    """Test BehaviourGenerator modifies trades for all 11 behavioral bias types."""
    rng = random.Random(42)
    market_gen = MarketGenerator(rng=rng)
    trade_sim = TradeSimulator(rng=rng)
    behaviour_gen = BehaviourGenerator(rng=rng)

    ctx = market_gen.generate_market(
        MarketScenarioConfig(
            scenario_id="s1", symbol="TSLA", regime=MarketRegime.RANGE_BOUND, seed=42
        )
    )
    base_trade = trade_sim.simulate_trade(ctx, user_id="u42")

    for bias_type in BiasType:
        bias_cfg = BiasInjectionConfig(
            bias_type=bias_type,
            intensity=0.8,
            description=f"Testing {bias_type.value}",
        )
        modified_trade = behaviour_gen.inject_bias(base_trade, bias_cfg)

        assert modified_trade.metadata["bias_applied"] == bias_type.value
        assert any(t.startswith("bias:") for t in modified_trade.tags)

        if bias_type == BiasType.REVENGE_TRADING:
            assert modified_trade.quantity > base_trade.quantity
        elif bias_type == BiasType.IGNORING_PLAN:
            assert modified_trade.initial_stop_loss is None
            assert modified_trade.initial_take_profit is None
