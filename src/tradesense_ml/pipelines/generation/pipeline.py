"""Concrete synthetic data generation pipeline orchestrator."""

import hashlib
import json
import random
from datetime import UTC, datetime
from typing import Any

from tradesense_ml.domain.schemas.coaching import CoachRequest
from tradesense_ml.domain.schemas.lineage import DatasetSplit, DatasetVersionMetadata
from tradesense_ml.domain.schemas.market_context import MarketContext, MarketRegime, VolatilityLevel
from tradesense_ml.domain.schemas.review import ReviewStage
from tradesense_ml.domain.schemas.synthetic import (
    BiasInjectionConfig,
    BiasType,
    MarketScenarioConfig,
    SyntheticGenerationBatch,
    SyntheticGeneratorConfig,
)
from tradesense_ml.domain.schemas.trade import Trade
from tradesense_ml.pipelines.generation.base import SyntheticGenerationPipeline
from tradesense_ml.pipelines.generation.behaviour_generator import BehaviourGenerator
from tradesense_ml.pipelines.generation.market_generator import MarketGenerator
from tradesense_ml.pipelines.generation.trade_simulator import TradeSimulator


class ConcreteSyntheticGenerationPipeline(SyntheticGenerationPipeline):
    """Concrete orchestrator for end-to-end deterministic synthetic market and trade data generation."""

    def __init__(self, generator_version: str = "0.1.0") -> None:
        """Initialize synthetic generation pipeline with generator version."""
        super().__init__()
        self.generator_version = generator_version

    def generate_dataset(
        self, config: SyntheticGeneratorConfig
    ) -> tuple[list[CoachRequest], DatasetVersionMetadata]:
        """Generate synthetic dataset of CoachRequest objects with full lineage provenance."""
        rng = random.Random(config.seed)

        market_gen = MarketGenerator(rng=rng)
        behaviour_gen = BehaviourGenerator(rng=rng)
        trade_sim = TradeSimulator(rng=rng)

        # Regimes & Biases sampling lists
        regimes = (
            list(config.market_regime_distribution.keys())
            if config.market_regime_distribution
            else list(MarketRegime)
        )
        regime_weights = (
            list(config.market_regime_distribution.values())
            if config.market_regime_distribution
            else [1.0] * len(regimes)
        )

        biases = (
            list(config.behaviour_probabilities.keys())
            if config.behaviour_probabilities
            else list(BiasType)
        )
        bias_weights = (
            list(config.behaviour_probabilities.values())
            if config.behaviour_probabilities
            else [1.0] * len(biases)
        )

        risk_profiles = config.risk_profiles or ["conservative", "moderate", "aggressive"]

        requests: list[CoachRequest] = []

        for i in range(config.num_samples):
            # 1. Sample regime, bias, risk profile
            chosen_regime: MarketRegime = rng.choices(regimes, weights=regime_weights, k=1)[0]
            chosen_bias: BiasType = rng.choices(biases, weights=bias_weights, k=1)[0]
            chosen_risk: str = rng.choice(risk_profiles)

            # 2. Generate Market Context
            scenario_cfg = MarketScenarioConfig(
                scenario_id=f"scen_{i+1:04d}",
                symbol=rng.choice(["AAPL", "BTC/USD", "ETH/USD", "NVDA", "TSLA", "EUR/USD"]),
                regime=chosen_regime,
                volatility=rng.choice(list(VolatilityLevel)),
                num_candles=100,
                seed=rng.randint(1, 10_000_000),
            )

            market_ctx = market_gen.generate_market(scenario_cfg)

            # 3. Simulate Base Trade
            trade = trade_sim.simulate_trade_with_outcome(
                context=market_ctx,
                user_id=f"trader_{(i % 10) + 1:03d}",
                risk_profile=chosen_risk,
                trade_id=f"trd_{i+1:04d}",
            )

            # 4. Inject Behavioural Bias
            bias_cfg = BiasInjectionConfig(
                bias_type=chosen_bias,
                intensity=round(rng.uniform(0.3, 0.95), 2),
                description=f"Synthetic bias injection of {chosen_bias.value}",
            )
            final_trade = behaviour_gen.inject_bias(trade, bias_cfg)

            # 5. Build CoachRequest
            coach_req = CoachRequest(
                request_id=f"req_{i+1:04d}",
                user_id=final_trade.user_id,
                trade=final_trade,
                market_context=market_ctx,
                user_notes=f"Trade taken under {chosen_regime.value} market context.",
                requested_aspects=["risk", "discipline", "general"],
            )
            requests.append(coach_req)

        # Build lineage provenance metadata
        config_json = json.dumps(config.model_dump(), sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        lineage = DatasetVersionMetadata(
            dataset_id=f"synthetic_trades_{config.seed}",
            dataset_version="0.1.0",
            teacher_model="synthetic-generator-v0.1",
            prompt_version="v1",
            rubric_version="v1",
            generator_version=self.generator_version,
            review_version="v1",
            generation_timestamp=datetime.now(UTC),
            source_hash=config_hash,
            review_status=ReviewStage.AUTOMATED_VALIDATION,
            sample_count=len(requests),
            split=DatasetSplit.TRAIN,
            metadata={
                "seed": config.seed,
                "config_hash": config_hash,
                "requested_samples": config.num_samples,
                "output_format": config.output_format,
            },
        )

        return requests, lineage

    def run(
        self, input_data: SyntheticGenerationBatch, **kwargs: Any
    ) -> list[tuple[Trade, MarketContext]]:
        """Run batch generation of market scenarios and trades for abstract pipeline compliance."""
        seed = input_data.generation_metadata.get("seed", 42)
        cfg = SyntheticGeneratorConfig(
            num_samples=input_data.total_samples,
            seed=seed,
        )
        requests, _ = self.generate_dataset(cfg)
        return [
            (req.trade, req.market_context) for req in requests if req.market_context is not None
        ]
