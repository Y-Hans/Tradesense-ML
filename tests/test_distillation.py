"""Comprehensive unit and integration test suite for the TradeSense ML Distillation Pipeline milestone."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradesense_ml.cli.main import app as cli_app
from tradesense_ml.config.settings import DistillationSettings
from tradesense_ml.distillation.curriculum import CurriculumBuilder, CurriculumStrategyRegistry
from tradesense_ml.distillation.exporters import DistillationExporterManager
from tradesense_ml.distillation.filtering import FilteringEngine
from tradesense_ml.distillation.lineage import DistillationLineageTracker
from tradesense_ml.distillation.pipeline import DistillationPipeline
from tradesense_ml.distillation.preference import PreferenceBuilder
from tradesense_ml.distillation.reporting import DistillationReportGenerator
from tradesense_ml.distillation.runner import DistillationRunner
from tradesense_ml.distillation.sampling import SamplingEngine, SamplingStrategyRegistry
from tradesense_ml.distillation.selection import SelectionEngine, SelectionStrategyRegistry
from tradesense_ml.distillation.statistics import StatisticsGenerator
from tradesense_ml.distillation.strategies import DistillationStrategyRegistry
from tradesense_ml.distillation.validation import DistillationValidator
from tradesense_ml.domain.schemas.benchmark import (
    BenchmarkArtifact,
    BenchmarkLineage,
    BenchmarkMetadata,
    BenchmarkProfile,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkScore,
    BenchmarkSummary,
)
from tradesense_ml.domain.schemas.dataset import (
    DatasetArtifact,
    DatasetExample,
    DatasetLineage,
    DatasetManifest,
    DatasetMetadata,
    DatasetStatistics,
)
from tradesense_ml.domain.schemas.distillation import (
    CurriculumStage,
    DistillationArtifact,
    DistillationExample,
    DistillationLineage,
    DistillationProcessingResult,
    PreferencePair,
)

cli_runner = CliRunner()


@pytest.fixture
def mock_dataset_artifact() -> DatasetArtifact:
    """Fixture providing a deterministic DatasetArtifact release."""
    examples = [
        DatasetExample(
            example_id=f"ex_{i:03d}",
            instruction="Provide structured coaching on this trade execution.",
            input=f"Trade #{i}: Long BTCUSDT @ 65000, Stop Loss @ 64000. Risk-reward 2.5:1.",
            output=(
                f"Trade Coaching Feedback #{i}:\n"
                "1. Risk Analysis: Stop loss and position size adhere strictly to risk limits.\n"
                "2. Discipline Analysis: Excellent emotional control and rule adherence.\n"
                "3. Actionability: Next time, consider scaling out at key resistance levels."
            ),
            prompt="System prompt: You are an expert trading coach.",
            reasoning="Reasoning step: Trader followed rules and maintained stop loss.",
            review_info={"quality_score": 8.5 + (i % 3) * 0.5, "verdict": "approved"},
            lineage={"generator": "synthetic_v1"},
            metadata={"difficulty": 0.1 * i, "teacher_id": "teacher_llm_v1"},
        )
        for i in range(10)
    ]

    return DatasetArtifact(
        artifact_id="test_dataset_v1",
        dataset_metadata=DatasetMetadata(
            name="test_dataset_v1",
            description="Test dataset release",
            version="v1.0.0",
        ),
        lineage=DatasetLineage(
            dataset_id="test_dataset_v1",
            dataset_version="v1.0.0",
            configuration_hash="hash_12345",
        ),
        statistics=DatasetStatistics(
            dataset_id="test_dataset_v1",
            total_examples=10,
            approved_examples=10,
            split_sizes={"train": 8, "validation": 1, "test": 1},
        ),
        manifest=DatasetManifest(
            dataset_id="test_dataset_v1",
            version="v1.0.0",
            dataset_format="sft_instruction",
            split_sizes={"train": 8, "validation": 1, "test": 1},
            statistics_summary={"total": 10},
            lineage={},
            checksum="chk_123",
        ),
        splits={"train": examples[:8], "validation": [examples[8]], "test": [examples[9]]},
    )


@pytest.fixture
def mock_benchmark_artifact(mock_dataset_artifact: DatasetArtifact) -> BenchmarkArtifact:
    """Fixture providing a deterministic BenchmarkArtifact release."""
    return BenchmarkArtifact(
        artifact_id="test_benchmark_v1",
        metadata=BenchmarkMetadata(
            benchmark_id="test_benchmark_v1",
            name="Test Benchmark",
            suite_name="teacher_benchmark_suite",
            profile_name="teacher_evaluation",
            dataset_id=mock_dataset_artifact.artifact_id,
            dataset_version="v1.0.0",
        ),
        lineage=BenchmarkLineage(
            dataset_artifact_id=mock_dataset_artifact.artifact_id,
            dataset_version="v1.0.0",
            teacher_model="teacher_llm_v1",
            configuration_hash="bm_hash_123",
        ),
        profile=BenchmarkProfile(profile_id="teacher_evaluation", name="Teacher Evaluation"),
        suite_info={"case_count": 1, "pass_rate": 1.0},
        execution_results=[],
        results=[
            BenchmarkResult(
                case_id="coaching_quality",
                case_name="Coaching Quality",
                concern="Coaching Usefulness",
                passed=True,
                score=8.5,
            )
        ],
        metrics=[],
        scores=BenchmarkScore(
            overall_score=8.5, weighted_score=8.5, category_scores={"coaching": 8.5}
        ),
        summary=BenchmarkSummary(
            benchmark_id="test_benchmark_v1",
            overall_score=8.5,
            pass_rate=1.0,
            passed_cases=1,
            total_cases=1,
        ),
        report=BenchmarkReport(
            overall_score=8.5,
            category_scores={"coaching": 8.5},
            metric_breakdown=[],
            failures=[],
            ranking={"tier": "Gold"},
            configuration_summary={},
            dataset_summary={},
            model_summary={},
        ),
        configuration={},
        dataset_reference={"artifact_id": mock_dataset_artifact.artifact_id},
        model_reference={"teacher_model": "teacher_llm_v1"},
        prompt_reference={"prompt_version": "v1"},
    )


# 1. Test Domain Models
def test_distillation_domain_models() -> None:
    """Test instantiating distillation domain models."""
    ex = DistillationExample(
        example_id="ex_001",
        instruction="Instruction",
        input="Input",
        output="Output",
        prompt="Prompt",
        quality_score=9.0,
        difficulty=0.2,
        quality_tier="easy",
    )
    assert ex.example_id == "ex_001"
    assert ex.quality_score == 9.0

    pair = PreferencePair(
        pair_id="pair_001",
        example_id="ex_001",
        instruction="Inst",
        input="In",
        prompt="P",
        chosen_response="Chosen text",
        rejected_response="Rejected text",
        chosen_score=9.0,
        rejected_score=4.0,
    )
    assert pair.pair_id == "pair_001"
    assert pair.chosen_score > pair.rejected_score

    stage = CurriculumStage(
        stage_id="s1",
        name="Easy",
        stage_order=1,
        examples=[ex],
        example_ids=["ex_001"],
        example_count=1,
    )
    assert stage.example_count == 1


# 2. Test Selection Engine & Strategies
def test_selection_engine(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Selection Engine and pluggable strategies."""
    strategies = SelectionStrategyRegistry.list_strategies()
    assert "ThresholdSelection" in strategies
    assert "TopScoreSelection" in strategies
    assert "BalancedSelection" in strategies
    assert "WeightedSelection" in strategies

    engine = SelectionEngine()

    # ThresholdSelection
    res_thresh = engine.select(
        mock_dataset_artifact, strategy_name="ThresholdSelection", threshold=8.8
    )
    assert res_thresh.strategy_name == "ThresholdSelection"
    assert len(res_thresh.selected_examples) > 0

    # TopScoreSelection
    res_top = engine.select(mock_dataset_artifact, strategy_name="TopScoreSelection", top_k=3)
    assert len(res_top.selected_examples) == 3


# 3. Test Filtering Engine
def test_filtering_engine(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Filtering Engine rules and statistics."""
    engine = FilteringEngine()
    sel_res = SelectionEngine().select(mock_dataset_artifact)

    passed, rejected, stats = engine.filter_examples(
        sel_res.selected_examples, min_quality_score=8.0
    )
    assert len(passed) + len(rejected) == len(sel_res.selected_examples)
    assert "total_input" in stats
    assert "rejection_reasons" in stats


# 4. Test Sampling Engine & Strategies
def test_sampling_engine(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Sampling Engine and pluggable strategies."""
    strategies = SamplingStrategyRegistry.list_strategies()
    assert "UniformSampling" in strategies
    assert "WeightedSampling" in strategies
    assert "BalancedSampling" in strategies
    assert "CurriculumSampling" in strategies
    assert "RandomDeterministicSampling" in strategies

    engine = SamplingEngine()
    sel_res = SelectionEngine().select(mock_dataset_artifact)

    res_uni = engine.sample(
        sel_res.selected_examples, strategy_name="UniformSampling", sample_size=4
    )
    assert len(res_uni.sampled_examples) == 4

    res_curr = engine.sample(
        sel_res.selected_examples, strategy_name="CurriculumSampling", sample_size=3
    )
    assert len(res_curr.sampled_examples) == 3


# 5. Test Curriculum Builder & Strategies
def test_curriculum_builder(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Curriculum Builder and difficulty stages."""
    strategies = CurriculumStrategyRegistry.list_strategies()
    assert "StandardCurriculumStrategy" in strategies
    assert "DifficultyCurriculumStrategy" in strategies

    builder = CurriculumBuilder()
    sel_res = SelectionEngine().select(mock_dataset_artifact)

    stages = builder.build_curriculum(
        sel_res.selected_examples, strategy_name="StandardCurriculumStrategy"
    )
    assert len(stages) == 4
    assert stages[0].name == "Easy"
    assert stages[3].name == "Expert"


# 6. Test Preference Builder
def test_preference_builder(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Preference Builder generating PreferencePair objects."""
    builder = PreferenceBuilder()
    sel_res = SelectionEngine().select(mock_dataset_artifact)

    pairs = builder.build_preference_pairs(chosen_examples=sel_res.selected_examples)
    assert len(pairs) == len(sel_res.selected_examples)
    for p in pairs:
        assert isinstance(p, PreferencePair)
        assert p.chosen_score > p.rejected_score


# 7. Test Distillation Strategies & Processing Result
def test_distillation_strategies(
    mock_dataset_artifact: DatasetArtifact, mock_benchmark_artifact: BenchmarkArtifact
) -> None:
    """Test Distillation Strategies producing DistillationProcessingResult."""
    registered = DistillationStrategyRegistry.list_strategies()
    assert "SFTStrategy" in registered
    assert "DPOStrategy" in registered
    assert "ORPOStrategy" in registered
    assert "CurriculumStrategy" in registered
    assert "HybridStrategy" in registered

    sft_strat = DistillationStrategyRegistry.get("SFTStrategy")
    res_sft = sft_strat.execute(mock_dataset_artifact, mock_benchmark_artifact)
    assert isinstance(res_sft, DistillationProcessingResult)
    assert len(res_sft.sampled_examples) > 0

    dpo_strat = DistillationStrategyRegistry.get("DPOStrategy")
    res_dpo = dpo_strat.execute(mock_dataset_artifact, mock_benchmark_artifact)
    assert len(res_dpo.preference_pairs) > 0


# 8. Test Statistics & Lineage & Reporting & Validation
def test_pipeline_components(
    mock_dataset_artifact: DatasetArtifact, mock_benchmark_artifact: BenchmarkArtifact
) -> None:
    """Test Statistics, Lineage, Reporting, and Validation modules."""
    res = DistillationRunner().run_strategy(
        mock_dataset_artifact, mock_benchmark_artifact, strategy_name="HybridStrategy"
    )

    stats = StatisticsGenerator.generate_statistics(res)
    assert stats.total_examples == len(res.sampled_examples)

    lineage = DistillationLineageTracker.create_lineage(
        mock_dataset_artifact, mock_benchmark_artifact
    )
    assert isinstance(lineage, DistillationLineage)

    report = DistillationReportGenerator.generate_report(
        res, stats, mock_dataset_artifact, mock_benchmark_artifact
    )
    assert report.statistics == stats

    val_res = DistillationValidator.validate_distillation(
        mock_dataset_artifact, mock_benchmark_artifact
    )
    assert val_res.is_valid is True


# 9. Test Exporters
def test_distillation_exporters(
    mock_dataset_artifact: DatasetArtifact, mock_benchmark_artifact: BenchmarkArtifact
) -> None:
    """Test exporting DistillationArtifact to JSON, JSONL, Parquet, and Markdown."""
    pipeline = DistillationPipeline(default_strategy="HybridStrategy")
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact = pipeline.run(
            input_data=mock_dataset_artifact,
            benchmark_artifact=mock_benchmark_artifact,
            output_dir=tmp_dir,
        )

        descriptors = DistillationExporterManager.export_artifact(
            artifact=artifact,
            output_dir=tmp_dir,
            formats=["json", "jsonl", "parquet", "md"],
        )

        assert len(descriptors) == 4
        for d in descriptors:
            assert Path(d["path"]).exists()
            assert d["size_bytes"] > 0


# 10. Test End-to-End Pipeline Execution
def test_distillation_pipeline_e2e(
    mock_dataset_artifact: DatasetArtifact, mock_benchmark_artifact: BenchmarkArtifact
) -> None:
    """Test end-to-end flow: DatasetArtifact + BenchmarkArtifact -> DistillationPipeline -> DistillationArtifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = DistillationPipeline(default_strategy="HybridStrategy")
        artifact = pipeline.run(
            input_data=mock_dataset_artifact,
            benchmark_artifact=mock_benchmark_artifact,
            distillation_id="dist_e2e_001",
            output_dir=tmp_dir,
        )

        assert isinstance(artifact, DistillationArtifact)
        assert artifact.artifact_id == "dist_e2e_001"
        assert artifact.summary.total_sampled_examples > 0
        assert artifact.summary.total_preference_pairs > 0
        assert len(artifact.export_files) > 0


# 11. Test Hydra Configuration Model
def test_distillation_settings() -> None:
    """Test DistillationSettings model."""
    settings = DistillationSettings()
    assert settings.distillation_strategy == "SFTStrategy"
    assert settings.selection_threshold == 7.0


# 12. Test CLI Commands
def test_distillation_cli(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Typer CLI subcommands tsml distillation run, report, validate, export, list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_file = Path(tmp_dir) / "dataset.json"
        ds_file.write_text(mock_dataset_artifact.model_dump_json(), encoding="utf-8")

        # Test run
        res = cli_runner.invoke(
            cli_app,
            ["distillation", "run", "--dataset", str(ds_file), "--output-dir", tmp_dir],
        )
        assert res.exit_code == 0
        assert "Distillation Execution Completed Successfully!" in res.stdout

        # Test list
        res_list = cli_runner.invoke(cli_app, ["distillation", "list"])
        assert res_list.exit_code == 0
        assert "SFTStrategy" in res_list.stdout

        # Test report
        dist_file = (
            Path(tmp_dir)
            / "tradesense_distillation_v1"
            / "tradesense_distillation_v1_distillation.json"
        )

        res_rpt = cli_runner.invoke(cli_app, ["distillation", "report", str(dist_file)])
        assert res_rpt.exit_code == 0
        assert "Mean Quality Score:" in res_rpt.stdout

        # Test validate
        res_val = cli_runner.invoke(cli_app, ["distillation", "validate", str(dist_file)])
        assert res_val.exit_code == 0
        assert "is valid" in res_val.stdout

        # Test export
        res_exp = cli_runner.invoke(
            cli_app, ["distillation", "export", str(dist_file), "--output-dir", tmp_dir]
        )
        assert res_exp.exit_code == 0
        assert "Successfully exported" in res_exp.stdout
