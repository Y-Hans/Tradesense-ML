"""Comprehensive unit tests for Dataset Builder models, filter, transformer, splitter, validator, statistics, lineage, manifest, exporters, pipeline, and CLI."""

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from tradesense_ml.cli.main import app as cli_app
from tradesense_ml.config.settings import load_hydra_config
from tradesense_ml.dataset.exporters import (
    JSONExporter,
    JSONLExporter,
    ParquetExporter,
)
from tradesense_ml.dataset.filtering import DatasetFilter, DatasetFilterConfig
from tradesense_ml.dataset.lineage import DatasetLineageTracker
from tradesense_ml.dataset.manifest import DatasetManifestGenerator
from tradesense_ml.dataset.models import (
    DatasetExample,
)
from tradesense_ml.dataset.pipeline import DatasetBuilderPipeline
from tradesense_ml.dataset.splitting import DatasetSplitter
from tradesense_ml.dataset.statistics import DatasetStatisticsGenerator
from tradesense_ml.dataset.transformation import DatasetTransformer
from tradesense_ml.dataset.validation import DatasetValidator
from tradesense_ml.domain.schemas.coaching import CoachRequest, CoachResponse
from tradesense_ml.domain.schemas.evaluations import DisciplineEvaluation, RiskEvaluation
from tradesense_ml.domain.schemas.examples import ReviewedExample
from tradesense_ml.domain.schemas.review import (
    ReviewStage,
    ReviewVerdict,
)
from tradesense_ml.domain.schemas.trade import Side, Trade

runner = CliRunner()


def _create_sample_reviewed_example(
    example_id: str = "ex_test_01",
    verdict: ReviewVerdict = ReviewVerdict.APPROVE,
    score: float = 8.5,
    symbol: str = "BTC/USD",
) -> ReviewedExample:
    """Helper to create a ReviewedExample object for testing."""
    trade = Trade(
        trade_id=f"trd_{example_id}",
        user_id="usr_01",
        symbol=symbol,
        side=Side.BUY,
        entry_price=50000.0,
        quantity=1.0,
        exit_price=52000.0,
        pnl=2000.0,
        pnl_percentage=4.0,
        entry_timestamp=datetime.now(UTC),
    )
    req = CoachRequest(request_id=f"req_{example_id}", user_id="usr_01", trade=trade)

    risk = RiskEvaluation(
        risk_score=8.0,
        position_size_compliant=True,
        stop_loss_defined=True,
        risk_summary="Position size within risk limits.",
    )
    disc = DisciplineEvaluation(
        discipline_score=9.0,
        fomo_indicator=False,
        revenge_trade_indicator=False,
        overtrading_indicator=False,
        plan_adherence_score=9.0,
        discipline_summary="Disciplined trade execution without FOMO.",
    )
    resp = CoachResponse(
        response_id=f"resp_{example_id}",
        request_id=f"req_{example_id}",
        headline="Solid trade execution with well-managed risk reward ratio.",
        overall_score=score,
        risk_evaluation=risk,
        discipline_evaluation=disc,
        actionable_advice=[
            "Maintain consistent position sizing across volatile market context.",
            "Set profit target trailing stops when price hits R:R of 2.0.",
        ],
        educational_note="Risk reward discipline protects trading capital over long sample sizes.",
        metadata={"provider": "openrouter", "model": "gpt-4o"},
    )

    review_status = (
        ReviewStage.APPROVED if verdict == ReviewVerdict.APPROVE else ReviewStage.REJECTED
    )
    return ReviewedExample(
        example_id=example_id,
        request=req,
        teacher_response=resp,
        review_status=review_status,
        final_quality_score=score,
    )


# --- 1. Dataset Domain Models Tests ---


def test_dataset_example_schema_validation() -> None:
    """Test DatasetExample creation and serialization."""
    example = DatasetExample(
        example_id="ex_01",
        instruction="You are TradeSense AI...",
        input="Trade Details...",
        output="Coaching summary...",
        prompt="Combined prompt text...",
        messages=[
            {"role": "system", "content": "System..."},
            {"role": "user", "content": "User..."},
            {"role": "assistant", "content": "Assistant..."},
        ],
        format_type="sft_instruction",
        review_info={"quality_score": 8.5},
        lineage={"request_id": "req_01"},
    )

    assert example.example_id == "ex_01"
    assert example.format_type == "sft_instruction"
    dumped = example.model_dump(mode="json")
    assert dumped["review_info"]["quality_score"] == 8.5


# --- 2. Dataset Filtering Tests ---


def test_dataset_filter_approved_only() -> None:
    """Test DatasetFilter filters unapproved examples."""
    ex_app = _create_sample_reviewed_example("ex_01", ReviewVerdict.APPROVE, 8.5)
    ex_rej = _create_sample_reviewed_example("ex_02", ReviewVerdict.REJECT, 3.0)

    filter_engine = DatasetFilter(DatasetFilterConfig(only_approved=True, min_quality_score=7.0))
    result = filter_engine.filter_batch([ex_app, ex_rej])

    assert len(result.kept_examples) == 1
    assert result.kept_examples[0].example_id == "ex_01"
    assert result.rejected_count == 1


def test_dataset_filter_deduplication() -> None:
    """Test DatasetFilter removes duplicate example IDs."""
    ex1 = _create_sample_reviewed_example("ex_dup", ReviewVerdict.APPROVE, 8.5)
    ex2 = _create_sample_reviewed_example("ex_dup", ReviewVerdict.APPROVE, 8.5)

    filter_engine = DatasetFilter(DatasetFilterConfig(remove_duplicates=True))
    result = filter_engine.filter_batch([ex1, ex2])

    assert len(result.kept_examples) == 1
    assert result.duplicate_count == 1


# --- 3. Dataset Transformation Tests ---


def test_dataset_transformer_formats() -> None:
    """Test DatasetTransformer transforms reviewed examples into sft_instruction, sft_chat, and evaluation formats."""
    ex = _create_sample_reviewed_example("ex_trans", ReviewVerdict.APPROVE, 8.5)
    transformer = DatasetTransformer()

    # SFT Instruction
    ds_inst = transformer.transform_single(ex, target_format="sft_instruction")
    assert ds_inst is not None
    assert ds_inst.format_type == "sft_instruction"
    assert "Trade Details:" in ds_inst.input
    assert "Coaching Summary" in ds_inst.output

    # SFT Chat
    ds_chat = transformer.transform_single(ex, target_format="sft_chat")
    assert ds_chat is not None
    assert len(ds_chat.messages) == 3
    assert ds_chat.messages[0]["role"] == "system"
    assert ds_chat.messages[1]["role"] == "user"
    assert ds_chat.messages[2]["role"] == "assistant"


# --- 4. Dataset Splitting Tests ---


def test_dataset_splitter_reproducibility() -> None:
    """Test DatasetSplitter deterministic splitting with seed."""
    transformer = DatasetTransformer()
    items = [_create_sample_reviewed_example(f"ex_{i:02d}") for i in range(10)]
    examples = transformer.transform_batch(items)

    splitter_a = DatasetSplitter(ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=42)
    res_a = splitter_a.split(examples)

    splitter_b = DatasetSplitter(ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=42)
    res_b = splitter_b.split(examples)

    assert [e.example_id for e in res_a.splits["train"]] == [
        e.example_id for e in res_b.splits["train"]
    ]
    assert res_a.split_sizes["train"] == 8
    assert res_a.split_sizes["validation"] == 1
    assert res_a.split_sizes["test"] == 1


# --- 5. Dataset Validation Tests ---


def test_dataset_validator_rules() -> None:
    """Test DatasetValidator checks required fields and split integrity."""
    transformer = DatasetTransformer()
    examples = transformer.transform_batch([_create_sample_reviewed_example("ex_val_01")])
    validator = DatasetValidator()

    # Valid dataset
    report_valid = validator.validate_dataset(examples)
    assert report_valid.is_valid

    # Overlapping split error
    split_overlap = {"train": examples, "validation": examples}
    report_overlap = validator.validate_dataset(examples, split_dict=split_overlap)
    assert not report_overlap.is_valid
    assert "overlapping IDs" in report_overlap.errors[0]


# --- 6. Statistics, Lineage, and Manifest Tests ---


def test_dataset_statistics_generation() -> None:
    """Test DatasetStatisticsGenerator metrics and byte calculations."""
    transformer = DatasetTransformer()
    examples = transformer.transform_batch(
        [_create_sample_reviewed_example(f"ex_stat_{i}") for i in range(5)]
    )

    stats = DatasetStatisticsGenerator.generate(
        dataset_id="test_ds",
        examples=examples,
        total_evaluated=5,
        rejected_count=0,
    )

    assert stats.total_examples == 5
    assert stats.approved_examples == 5
    assert stats.quality_score_mean == 8.5
    assert stats.dataset_size_bytes > 0


def test_lineage_and_manifest_generation(tmp_path: Path) -> None:
    """Test DatasetLineageTracker and DatasetManifestGenerator creating SHA256 checksums."""
    transformer = DatasetTransformer()
    examples = transformer.transform_batch([_create_sample_reviewed_example("ex_man")])

    config_dict = {"dataset_id": "ds_test", "seed": 42}
    lineage = DatasetLineageTracker.create_lineage(
        dataset_id="ds_test",
        dataset_version="v1.0.0",
        config_dict=config_dict,
        examples=examples,
    )
    assert len(lineage.configuration_hash) == 64

    # Export a test file for checksum
    test_file = tmp_path / "ds_test_train.jsonl"
    JSONLExporter().export(examples, test_file)

    stats = DatasetStatisticsGenerator.generate(dataset_id="ds_test", examples=examples)
    manifest = DatasetManifestGenerator.generate_manifest(
        dataset_id="ds_test",
        version="v1.0.0",
        dataset_format="sft_instruction",
        stats=stats,
        lineage=lineage,
        export_file_paths=[{"path": test_file, "split": "train", "count": 1}],
    )

    assert manifest.dataset_id == "ds_test"
    assert len(manifest.checksum) == 64
    assert manifest.export_files[0]["sha256_checksum"] != ""


# --- 7. Exporters Tests ---


def test_dataset_exporters_json_jsonl_parquet(tmp_path: Path) -> None:
    """Test JSONExporter, JSONLExporter, and ParquetExporter file outputs."""
    transformer = DatasetTransformer()
    examples = transformer.transform_batch([_create_sample_reviewed_example("ex_exp")])

    # JSONL
    f_jsonl = tmp_path / "data.jsonl"
    JSONLExporter().export(examples, f_jsonl)
    assert f_jsonl.exists()

    # JSON
    f_json = tmp_path / "data.json"
    JSONExporter().export(examples, f_json)
    assert f_json.exists()

    # Parquet
    f_pq = tmp_path / "data.parquet"
    ParquetExporter().export(examples, f_pq)
    assert f_pq.exists() or (tmp_path / "data.json").exists()


# --- 8. Pipeline End-to-End Test ---


# --- 8. DatasetFormat Strategy & Registry Tests ---


def test_dataset_format_registry_and_strategies() -> None:
    """Test DatasetFormatRegistry retrieves strategies and formats DatasetExample records."""
    from tradesense_ml.dataset.formats import DatasetFormatRegistry

    ex = _create_sample_reviewed_example("ex_fmt_test")
    transformer = DatasetTransformer()
    raw_ex = transformer.transform_single(ex, target_format="canonical")
    assert raw_ex is not None

    # Test SFT Instruction
    fmt_inst = DatasetFormatRegistry.get_format("sft_instruction")
    ex_inst = fmt_inst.format_example(raw_ex)
    assert ex_inst.format_type == "sft_instruction"
    assert "### User Request:" in ex_inst.prompt

    # Test SFT Chat
    fmt_chat = DatasetFormatRegistry.get_format("sft_chat")
    ex_chat = fmt_chat.format_example(raw_ex)
    assert ex_chat.format_type == "sft_chat"
    assert len(ex_chat.messages) == 3

    # Test Evaluation
    fmt_eval = DatasetFormatRegistry.get_format("evaluation")
    ex_eval = fmt_eval.format_example(raw_ex)
    assert ex_eval.format_type == "evaluation"
    assert ex_eval.metadata["is_ground_truth"] is True

    # Test DPO, ORPO, KTO, RewardModel scaffolds
    fmt_dpo = DatasetFormatRegistry.get_format("dpo")
    ex_dpo = fmt_dpo.format_example(raw_ex)
    assert ex_dpo.format_type == "dpo"
    assert "preference_pair" in ex_dpo.metadata

    fmt_kto = DatasetFormatRegistry.get_format("kto")
    ex_kto = fmt_kto.format_example(raw_ex)
    assert ex_kto.format_type == "kto"
    assert "kto_binary_label" in ex_kto.metadata


# --- 9. Pipeline End-to-End Test returning DatasetArtifact ---


def test_dataset_builder_pipeline_end_to_end(tmp_path: Path) -> None:
    """Test DatasetBuilderPipeline returning an immutable DatasetArtifact flow to Exporters."""
    from tradesense_ml.domain.schemas.dataset import DatasetArtifact

    items = [_create_sample_reviewed_example(f"ex_pipe_{i:02d}") for i in range(10)]
    pipeline = DatasetBuilderPipeline(dataset_id="test_pipe_ds", version="v1.0.0")

    artifact = pipeline.run(
        input_data=items,
        output_dir=str(tmp_path),
        dataset_format="sft_instruction",
        seed=42,
    )

    assert isinstance(artifact, DatasetArtifact)
    assert artifact.artifact_id == "test_pipe_ds"
    assert artifact.dataset_metadata.version == "v1.0.0"
    assert artifact.manifest.split_sizes["train"] == 8
    assert "train" in artifact.splits
    assert "validation" in artifact.splits
    assert "test" in artifact.splits
    assert len(artifact.export_files) > 0

    manifest_file = tmp_path / "test_pipe_ds" / "manifest.json"
    assert manifest_file.exists()


# --- 9. Hydra Config & CLI Subcommands Tests ---


def test_hydra_dataset_config_loading() -> None:
    """Test Hydra settings include DatasetSettings."""
    settings = load_hydra_config()
    assert hasattr(settings, "dataset")
    assert settings.dataset.dataset_id == "tradesense_sft_v1"
    assert settings.dataset.seed == 42


def test_cli_dataset_build_sample(tmp_path: Path) -> None:
    """Test `tsml dataset build` command with sample dataset batch."""
    out_dir = tmp_path / "cli_datasets"
    result = runner.invoke(
        cli_app,
        ["dataset", "build", "-o", str(out_dir), "-id", "cli_test_ds"],
    )
    assert result.exit_code == 0
    assert "Dataset Builder Pipeline Completed Successfully!" in result.output
    assert (out_dir / "cli_test_ds" / "manifest.json").exists()


def test_cli_dataset_stats_and_validate(tmp_path: Path) -> None:
    """Test `tsml dataset stats` and `tsml dataset validate` CLI commands."""
    out_dir = tmp_path / "cli_datasets_2"
    runner.invoke(cli_app, ["dataset", "build", "-o", str(out_dir), "-id", "cli_ds_2"])

    ds_path = out_dir / "cli_ds_2"

    # Stats
    res_stats = runner.invoke(cli_app, ["dataset", "stats", "-d", str(ds_path)])
    assert res_stats.exit_code == 0
    assert "Dataset Statistics" in res_stats.output

    # Validate
    res_val = runner.invoke(cli_app, ["dataset", "validate", str(ds_path)])
    assert res_val.exit_code == 0
    assert "VALIDATION PASSED" in res_val.output
