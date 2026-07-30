"""Integration tests for Synthetic Generation Pipeline, seed determinism, and lineage."""

from tradesense_ml.domain.schemas.synthetic import SyntheticGeneratorConfig
from tradesense_ml.pipelines.generation.pipeline import ConcreteSyntheticGenerationPipeline


def test_synthetic_pipeline_generation() -> None:
    """Test full pipeline execution produces valid dataset and lineage metadata."""
    cfg = SyntheticGeneratorConfig(num_samples=15, seed=42)
    pipeline = ConcreteSyntheticGenerationPipeline()

    requests, lineage = pipeline.generate_dataset(cfg)

    assert len(requests) == 15
    assert lineage.sample_count == 15
    assert lineage.metadata["seed"] == 42
    assert lineage.source_hash is not None


def test_synthetic_pipeline_seed_reproducibility() -> None:
    """Test identical random seed produces 100% identical synthetic dataset outputs."""
    cfg1 = SyntheticGeneratorConfig(num_samples=10, seed=12345)
    cfg2 = SyntheticGeneratorConfig(num_samples=10, seed=12345)

    p1 = ConcreteSyntheticGenerationPipeline()
    p2 = ConcreteSyntheticGenerationPipeline()

    reqs1, line1 = p1.generate_dataset(cfg1)
    reqs2, line2 = p2.generate_dataset(cfg2)

    assert [r.model_dump() for r in reqs1] == [r.model_dump() for r in reqs2]
    assert line1.source_hash == line2.source_hash
