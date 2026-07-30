"""Unit tests for SyntheticDatasetValidator rules and error reporting."""

from tradesense_ml.domain.schemas.synthetic import SyntheticGeneratorConfig
from tradesense_ml.pipelines.generation.pipeline import ConcreteSyntheticGenerationPipeline
from tradesense_ml.pipelines.validation.synthetic_validator import SyntheticDatasetValidator


def test_validator_accepts_valid_batch() -> None:
    """Test validator approves clean generated synthetic samples."""
    cfg = SyntheticGeneratorConfig(num_samples=10, seed=42)
    pipeline = ConcreteSyntheticGenerationPipeline()
    requests, _ = pipeline.generate_dataset(cfg)

    validator = SyntheticDatasetValidator()
    all_valid, results = validator.validate_batch(requests)

    assert all_valid is True
    assert all(r.is_valid for r in results)


def test_validator_rejects_invalid_price() -> None:
    """Test validator rejects samples with negative or zero price."""
    cfg = SyntheticGeneratorConfig(num_samples=1, seed=42)
    pipeline = ConcreteSyntheticGenerationPipeline()
    requests, _ = pipeline.generate_dataset(cfg)

    invalid_dict = requests[0].model_dump(mode="json")
    invalid_dict["trade"]["entry_price"] = -10.0

    validator = SyntheticDatasetValidator()
    res = validator.validate_sample(invalid_dict)

    assert res.is_valid is False
    assert any("entry_price" in err.lower() or "entry price" in err.lower() for err in res.errors)
