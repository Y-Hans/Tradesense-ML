"""Unit tests for MarketGenerator across all 10 market regimes and seed determinism."""

import random

from tradesense_ml.domain.schemas.market_context import MarketRegime, VolatilityLevel
from tradesense_ml.domain.schemas.synthetic import MarketScenarioConfig
from tradesense_ml.pipelines.generation.market_generator import MarketGenerator


def test_market_generator_all_regimes() -> None:
    """Test MarketGenerator creates valid, consistent MarketContext for all 10 regimes."""
    market_gen = MarketGenerator(rng=random.Random(42))

    for regime in MarketRegime:
        cfg = MarketScenarioConfig(
            scenario_id=f"test_{regime.value}",
            symbol="BTC/USD",
            regime=regime,
            volatility=VolatilityLevel.HIGH,
            seed=123,
        )
        context = market_gen.generate_market(cfg)

        assert context.regime == regime
        assert context.symbol == "BTC/USD"
        assert context.context_id == f"ctx_test_{regime.value}"
        assert len(context.support_levels) >= 1
        assert len(context.resistance_levels) >= 1
        assert context.indicators.rsi_14 is not None
        assert 0 <= context.indicators.rsi_14 <= 100
        assert context.indicators.macd_histogram is not None
        assert context.indicators.atr_14 is not None and context.indicators.atr_14 > 0
        assert context.overall_trend_score is not None
        assert -1.0 <= context.overall_trend_score <= 1.0


def test_market_generator_determinism() -> None:
    """Test that identical seeds yield identical MarketContext outputs."""
    gen1 = MarketGenerator()
    gen2 = MarketGenerator()

    cfg1 = MarketScenarioConfig(
        scenario_id="s1", symbol="AAPL", regime=MarketRegime.BULLISH_TREND, seed=999
    )
    cfg2 = MarketScenarioConfig(
        scenario_id="s1", symbol="AAPL", regime=MarketRegime.BULLISH_TREND, seed=999
    )

    ctx1 = gen1.generate_market(cfg1)
    ctx2 = gen2.generate_market(cfg2)

    assert ctx1.model_dump() == ctx2.model_dump()
