"""Trade outcome generator modeling plausible outcomes based on regime, behavior, and risk profile."""

import random
from enum import Enum

from tradesense_ml.domain.schemas.market_context import MarketContext, MarketRegime
from tradesense_ml.domain.schemas.synthetic import BiasType


class TradeOutcome(str, Enum):
    """Plausible synthetic trade outcome classifications."""

    DISCIPLINED_WINNER = "disciplined_winner"
    DISCIPLINED_LOSER = "disciplined_loser"
    LUCKY_WINNER = "lucky_winner"
    RECKLESS_WINNER = "reckless_winner"
    REVENGE_LOSS = "revenge_loss"
    FOMO_LOSS = "FOMO_loss"
    OVERLEVERAGED_LIQUIDATION = "overleveraged_liquidation"
    PREMATURE_EXIT = "premature_exit"
    MISSED_OPPORTUNITY = "missed_opportunity"


class TradeOutcomeGenerator:
    """Outcome generator determining outcome probabilities given regime, psychology, and risk profile."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Initialize outcome generator with optional random number generator instance."""
        self._rng = rng or random.Random(42)

    def determine_outcome(
        self,
        market_context: MarketContext,
        bias_type: BiasType,
        risk_profile: str = "moderate",
    ) -> TradeOutcome:
        """Select a plausible trade outcome based on market regime, trader behavior, and risk profile."""
        rng = self._rng
        regime = market_context.regime

        # Base probabilities
        outcomes: list[TradeOutcome] = list(TradeOutcome)
        weights = [1.0] * len(outcomes)

        # Modify weights according to bias
        if bias_type == BiasType.DISCIPLINE or bias_type == BiasType.NO_BIAS_CLEAN:
            weights[outcomes.index(TradeOutcome.DISCIPLINED_WINNER)] += 4.0
            weights[outcomes.index(TradeOutcome.DISCIPLINED_LOSER)] += 3.0
            weights[outcomes.index(TradeOutcome.REVENGE_LOSS)] *= 0.1
            weights[outcomes.index(TradeOutcome.FOMO_LOSS)] *= 0.1

        elif bias_type in (BiasType.FOMO, BiasType.CHASING_PRICE):
            weights[outcomes.index(TradeOutcome.FOMO_LOSS)] += 5.0
            weights[outcomes.index(TradeOutcome.RECKLESS_WINNER)] += 2.0
            weights[outcomes.index(TradeOutcome.MISSED_OPPORTUNITY)] += 1.5

        elif bias_type == BiasType.REVENGE_TRADING:
            weights[outcomes.index(TradeOutcome.REVENGE_LOSS)] += 6.0
            weights[outcomes.index(TradeOutcome.OVERLEVERAGED_LIQUIDATION)] += 3.0

        elif bias_type == BiasType.OVERCONFIDENCE:
            weights[outcomes.index(TradeOutcome.RECKLESS_WINNER)] += 3.0
            weights[outcomes.index(TradeOutcome.OVERLEVERAGED_LIQUIDATION)] += 4.0

        elif bias_type in (BiasType.FEAR, BiasType.HESITATION):
            weights[outcomes.index(TradeOutcome.PREMATURE_EXIT)] += 5.0
            weights[outcomes.index(TradeOutcome.MISSED_OPPORTUNITY)] += 4.0

        # Modify weights according to market regime
        if regime in (MarketRegime.FLASH_CRASH, MarketRegime.BEARISH_BREAKDOWN):
            weights[outcomes.index(TradeOutcome.OVERLEVERAGED_LIQUIDATION)] += 3.0
            weights[outcomes.index(TradeOutcome.DISCIPLINED_LOSER)] += 2.0

        elif regime in (MarketRegime.BULLISH_TREND, MarketRegime.HIGH_VOLATILITY_BREAKOUT):
            weights[outcomes.index(TradeOutcome.DISCIPLINED_WINNER)] += 2.5
            weights[outcomes.index(TradeOutcome.LUCKY_WINNER)] += 2.0

        # Modify weights according to risk profile
        if risk_profile == "aggressive":
            weights[outcomes.index(TradeOutcome.OVERLEVERAGED_LIQUIDATION)] += 2.5
            weights[outcomes.index(TradeOutcome.RECKLESS_WINNER)] += 2.0
        elif risk_profile == "conservative":
            weights[outcomes.index(TradeOutcome.PREMATURE_EXIT)] += 2.0
            weights[outcomes.index(TradeOutcome.DISCIPLINED_LOSER)] += 1.5

        chosen_outcome: TradeOutcome = rng.choices(outcomes, weights=weights, k=1)[0]
        return chosen_outcome
