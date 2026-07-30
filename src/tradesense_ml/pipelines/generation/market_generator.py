"""Market context and environment generator implementation."""

import random
from typing import Any

from tradesense_ml.domain.schemas.market_context import (
    MarketContext,
    MarketRegime,
    TechnicalIndicators,
    VolatilityLevel,
)
from tradesense_ml.domain.schemas.synthetic import MarketScenarioConfig
from tradesense_ml.pipelines.generation.base import BaseMarketGenerator


class MarketGenerator(BaseMarketGenerator):
    """Synthetic market context generator producing realistic market environments."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Initialize market generator with optional random number generator instance."""
        self._rng = rng or random.Random(42)

    def generate_market(self, config: MarketScenarioConfig) -> MarketContext:
        """Generate realistic synthetic market context according to configuration."""
        rng = random.Random(config.seed) if config.seed is not None else self._rng
        regime = config.regime
        symbol = config.symbol
        scenario_id = config.scenario_id

        base_price = rng.uniform(50.0, 500.0)

        # Determine regime parameters
        if regime == MarketRegime.BULLISH_TREND:
            trend_score = rng.uniform(0.5, 0.95)
            volatility = config.volatility if config.volatility else VolatilityLevel.MEDIUM
            rsi = rng.uniform(55.0, 78.0)
            macd_hist = rng.uniform(0.5, 4.5)
            sma_50_above_200 = True
            news_sentiment = rng.uniform(0.2, 0.8)
            support_pcts = [-0.02, -0.05, -0.08]
            resistance_pcts = [0.03, 0.07, 0.12]
            avg_volume = rng.uniform(1_000_000, 5_000_000)
            volume_spike_ratio = rng.uniform(1.0, 1.5)
            liquidity = rng.uniform(0.7, 0.95)
            regime_confidence = rng.uniform(0.8, 0.98)

        elif regime == MarketRegime.BEARISH_TREND:
            trend_score = rng.uniform(-0.95, -0.5)
            volatility = config.volatility if config.volatility else VolatilityLevel.HIGH
            rsi = rng.uniform(22.0, 45.0)
            macd_hist = rng.uniform(-4.5, -0.5)
            sma_50_above_200 = False
            news_sentiment = rng.uniform(-0.8, -0.2)
            support_pcts = [-0.04, -0.09, -0.15]
            resistance_pcts = [0.02, 0.05, 0.08]
            avg_volume = rng.uniform(1_500_000, 6_000_000)
            volume_spike_ratio = rng.uniform(1.2, 1.8)
            liquidity = rng.uniform(0.6, 0.9)
            regime_confidence = rng.uniform(0.78, 0.95)

        elif regime == MarketRegime.RANGE_BOUND:
            trend_score = rng.uniform(-0.15, 0.15)
            volatility = config.volatility if config.volatility else VolatilityLevel.LOW
            rsi = rng.uniform(42.0, 58.0)
            macd_hist = rng.uniform(-0.3, 0.3)
            sma_50_above_200 = rng.choice([True, False])
            news_sentiment = rng.uniform(-0.2, 0.2)
            support_pcts = [-0.015, -0.03]
            resistance_pcts = [0.015, 0.03]
            avg_volume = rng.uniform(500_000, 2_000_000)
            volume_spike_ratio = rng.uniform(0.8, 1.1)
            liquidity = rng.uniform(0.5, 0.8)
            regime_confidence = rng.uniform(0.7, 0.9)

        elif regime == MarketRegime.HIGH_VOLATILITY_BREAKOUT:
            trend_score = rng.uniform(0.4, 0.9)
            volatility = VolatilityLevel.HIGH
            rsi = rng.uniform(62.0, 85.0)
            macd_hist = rng.uniform(1.5, 6.0)
            sma_50_above_200 = True
            news_sentiment = rng.uniform(0.3, 0.9)
            support_pcts = [-0.01, -0.04]
            resistance_pcts = [0.05, 0.10, 0.18]
            avg_volume = rng.uniform(3_000_000, 10_000_000)
            volume_spike_ratio = rng.uniform(2.0, 4.5)
            liquidity = rng.uniform(0.65, 0.92)
            regime_confidence = rng.uniform(0.82, 0.96)

        elif regime == MarketRegime.BEARISH_BREAKDOWN:
            trend_score = rng.uniform(-0.9, -0.4)
            volatility = VolatilityLevel.HIGH
            rsi = rng.uniform(15.0, 35.0)
            macd_hist = rng.uniform(-6.0, -1.5)
            sma_50_above_200 = False
            news_sentiment = rng.uniform(-0.9, -0.3)
            support_pcts = [-0.05, -0.10, -0.18]
            resistance_pcts = [0.01, 0.04]
            avg_volume = rng.uniform(3_500_000, 12_000_000)
            volume_spike_ratio = rng.uniform(2.2, 5.0)
            liquidity = rng.uniform(0.5, 0.85)
            regime_confidence = rng.uniform(0.8, 0.95)

        elif regime == MarketRegime.HIGH_VOLATILITY:
            trend_score = rng.uniform(-0.3, 0.3)
            volatility = VolatilityLevel.EXTREME
            rsi = rng.uniform(30.0, 70.0)
            macd_hist = rng.uniform(-2.5, 2.5)
            sma_50_above_200 = rng.choice([True, False])
            news_sentiment = rng.uniform(-0.5, 0.5)
            support_pcts = [-0.05, -0.10]
            resistance_pcts = [0.05, 0.10]
            avg_volume = rng.uniform(4_000_000, 15_000_000)
            volume_spike_ratio = rng.uniform(1.8, 3.5)
            liquidity = rng.uniform(0.4, 0.75)
            regime_confidence = rng.uniform(0.65, 0.85)

        elif regime == MarketRegime.LOW_VOLATILITY_CONSOLIDATION:
            trend_score = rng.uniform(-0.1, 0.1)
            volatility = VolatilityLevel.LOW
            rsi = rng.uniform(46.0, 54.0)
            macd_hist = rng.uniform(-0.1, 0.1)
            sma_50_above_200 = True
            news_sentiment = rng.uniform(-0.1, 0.1)
            support_pcts = [-0.01, -0.02]
            resistance_pcts = [0.01, 0.02]
            avg_volume = rng.uniform(300_000, 1_200_000)
            volume_spike_ratio = rng.uniform(0.7, 1.0)
            liquidity = rng.uniform(0.7, 0.9)
            regime_confidence = rng.uniform(0.75, 0.92)

        elif regime == MarketRegime.NEWS_DRIVEN_SPIKE:
            trend_score = rng.choice([rng.uniform(0.6, 0.95), rng.uniform(-0.95, -0.6)])
            volatility = VolatilityLevel.EXTREME
            rsi = rng.uniform(75.0, 92.0) if trend_score > 0 else rng.uniform(8.0, 25.0)
            macd_hist = rng.uniform(3.0, 8.0) if trend_score > 0 else rng.uniform(-8.0, -3.0)
            sma_50_above_200 = trend_score > 0
            news_sentiment = rng.uniform(0.7, 1.0) if trend_score > 0 else rng.uniform(-1.0, -0.7)
            support_pcts = [-0.03, -0.08]
            resistance_pcts = [0.03, 0.08]
            avg_volume = rng.uniform(5_000_000, 20_000_000)
            volume_spike_ratio = rng.uniform(3.5, 8.0)
            liquidity = rng.uniform(0.3, 0.65)
            regime_confidence = rng.uniform(0.85, 0.98)

        elif regime == MarketRegime.FLASH_CRASH:
            trend_score = rng.uniform(-1.0, -0.85)
            volatility = VolatilityLevel.EXTREME
            rsi = rng.uniform(5.0, 18.0)
            macd_hist = rng.uniform(-12.0, -5.0)
            sma_50_above_200 = False
            news_sentiment = rng.uniform(-1.0, -0.8)
            support_pcts = [-0.08, -0.15, -0.25]
            resistance_pcts = [0.02, 0.06]
            avg_volume = rng.uniform(8_000_000, 30_000_000)
            volume_spike_ratio = rng.uniform(5.0, 12.0)
            liquidity = rng.uniform(0.15, 0.45)
            regime_confidence = rng.uniform(0.9, 0.99)

        else:  # MarketRegime.RECOVERY
            trend_score = rng.uniform(0.35, 0.75)
            volatility = VolatilityLevel.MEDIUM
            rsi = rng.uniform(48.0, 65.0)
            macd_hist = rng.uniform(0.2, 2.5)
            sma_50_above_200 = False
            news_sentiment = rng.uniform(0.1, 0.6)
            support_pcts = [-0.02, -0.06]
            resistance_pcts = [0.03, 0.07]
            avg_volume = rng.uniform(2_000_000, 6_000_000)
            volume_spike_ratio = rng.uniform(1.1, 1.9)
            liquidity = rng.uniform(0.6, 0.88)
            regime_confidence = rng.uniform(0.72, 0.91)

        atr_14 = base_price * (
            0.005
            if volatility == VolatilityLevel.LOW
            else (
                0.015
                if volatility == VolatilityLevel.MEDIUM
                else 0.035 if volatility == VolatilityLevel.HIGH else 0.07
            )
        )
        vwap_distance_pct = trend_score * 0.02

        indicators = TechnicalIndicators(
            rsi_14=round(rsi, 2),
            macd_histogram=round(macd_hist, 4),
            vwap_distance_pct=round(vwap_distance_pct, 4),
            atr_14=round(atr_14, 4),
            sma_50_above_200=sma_50_above_200,
        )

        support_levels = [round(base_price * (1.0 + p), 2) for p in support_pcts]
        resistance_levels = [round(base_price * (1.0 + p), 2) for p in resistance_pcts]

        metadata: dict[str, Any] = {
            "regime_confidence": round(regime_confidence, 4),
            "liquidity_score": round(liquidity, 4),
            "volume_profile": {
                "avg_volume": round(avg_volume, 0),
                "volume_spike_ratio": round(volume_spike_ratio, 2),
            },
            "ema_alignment": "bullish" if sma_50_above_200 else "bearish",
            "base_price": round(base_price, 2),
        }

        return MarketContext(
            context_id=f"ctx_{scenario_id}",
            symbol=symbol,
            regime=regime,
            volatility=volatility,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            indicators=indicators,
            news_sentiment_score=round(news_sentiment, 2),
            overall_trend_score=round(trend_score, 2),
            context_metadata=metadata,
        )
