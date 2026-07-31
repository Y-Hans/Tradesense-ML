"""Comprehensive unit and integration test suite for the TradeSense ML Benchmark Suite milestone."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradesense_ml.benchmark.cases import CaseRegistry
from tradesense_ml.benchmark.exporters import BenchmarkExporterManager
from tradesense_ml.benchmark.lineage import BenchmarkLineageTracker
from tradesense_ml.benchmark.metrics import MetricRegistry
from tradesense_ml.benchmark.pipeline import BenchmarkPipeline
from tradesense_ml.benchmark.profiles import ProfileRegistry
from tradesense_ml.benchmark.reporting import BenchmarkReportGenerator
from tradesense_ml.benchmark.runner import BenchmarkRunner
from tradesense_ml.benchmark.scoring import BenchmarkScoringEngine
from tradesense_ml.benchmark.suites import SuiteRegistry
from tradesense_ml.benchmark.validation import BenchmarkValidator
from tradesense_ml.cli.main import app as cli_app
from tradesense_ml.domain.schemas.benchmark import (
    BenchmarkArtifact,
    BenchmarkExecutionResult,
    BenchmarkLineage,
    BenchmarkMetadata,
    BenchmarkMetric,
    BenchmarkReport,
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
                "3. Actionability: Next time, consider scaling out at key resistance levels.\n"
                "Because of disciplined risk management, your expectancy remains positive."
            ),
            prompt="System prompt: You are an expert trading coach.",
            reasoning="Reasoning step: Trader followed rules and maintained stop loss.",
            review_info={"quality_score": 8.5 + (i % 3) * 0.5, "verdict": "approved"},
            lineage={"generator": "synthetic_v1"},
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


# 1. Test Domain Models
def test_benchmark_domain_models(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test instantiating and serializing canonical benchmark domain models."""
    profile = ProfileRegistry.get("teacher_evaluation")
    assert profile.profile_id == "teacher_evaluation"

    meta = BenchmarkMetadata(
        benchmark_id="bm_001",
        name="Test Benchmark",
        suite_name="teacher_benchmark_suite",
        profile_name="teacher_evaluation",
        dataset_id=mock_dataset_artifact.artifact_id,
        dataset_version="v1.0.0",
    )
    assert meta.benchmark_id == "bm_001"

    metric = BenchmarkMetric(
        metric_id="quality_score",
        name="Quality Score",
        metric_type="quality_score",
        value=8.5,
        unit="pts",
    )
    assert metric.value == 8.5


# 2. Test Metrics Registry & Metrics Calculation
def test_metric_registry_and_computations() -> None:
    """Test metric computation classes registered in MetricRegistry."""
    registered = MetricRegistry.list_metrics()
    expected_metrics = [
        "accuracy",
        "pass_rate",
        "quality_score",
        "consistency_score",
        "confidence",
        "latency",
        "token_usage",
        "cost",
        "response_length",
        "prompt_length",
    ]
    for em in expected_metrics:
        assert em in registered

    acc_metric = MetricRegistry.get("accuracy").compute([1, 1, 0, 1])
    assert acc_metric.value == 0.75

    pass_metric = MetricRegistry.get("pass_rate").compute([True, True, True])
    assert pass_metric.value == 1.0

    quality_metric = MetricRegistry.get("quality_score").compute([8.0, 9.0, 10.0])
    assert quality_metric.value == 9.0


# 3. Test Cases & Case Registry
def test_case_registry_and_case_evaluation(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test registered benchmark cases and un-scored evaluation execution."""
    cases = CaseRegistry.list_cases()
    assert len(cases) == 11

    case = CaseRegistry.get("coaching_quality")
    exec_res = case.evaluate(mock_dataset_artifact, suite_id="teacher_benchmark_suite")

    assert isinstance(exec_res, BenchmarkExecutionResult)
    assert exec_res.case_id == "coaching_quality"
    assert exec_res.total_items_evaluated == 10
    assert exec_res.status == "completed"
    assert "quality_score" in exec_res.raw_metrics


# 4. Test Suites & Suite Registry
def test_suite_registry_and_suite_execution(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test pluggable benchmark suites execution."""
    suites = SuiteRegistry.list_suites()
    assert "teacher_benchmark_suite" in suites
    assert "dataset_benchmark_suite" in suites
    assert "prompt_benchmark_suite" in suites

    suite = SuiteRegistry.get("teacher_benchmark_suite")
    raw_results = suite.run_cases(mock_dataset_artifact)
    assert len(raw_results) == 7


# 5. Test BenchmarkRunner
def test_benchmark_runner(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test BenchmarkRunner coordinating profile execution."""
    runner = BenchmarkRunner()
    profile = ProfileRegistry.get("teacher_evaluation")

    results = runner.run_profile(dataset_artifact=mock_dataset_artifact, profile=profile)
    assert len(results) == 7
    for res in results:
        assert isinstance(res, BenchmarkExecutionResult)


# 6. Test Scoring Engine
def test_benchmark_scoring_engine(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test BenchmarkScoringEngine converting raw measurements to scored results."""
    runner = BenchmarkRunner()
    profile = ProfileRegistry.get("teacher_evaluation")
    exec_results = runner.run_profile(mock_dataset_artifact, profile=profile)

    engine = BenchmarkScoringEngine()
    scored_results, metrics, scores, summary = engine.evaluate(exec_results, profile=profile)

    assert len(scored_results) == 7
    assert len(metrics) > 0
    assert isinstance(scores, BenchmarkScore)
    assert isinstance(summary, BenchmarkSummary)
    assert 0.0 <= scores.overall_score <= 10.0
    assert "coaching" in scores.category_scores


# 7. Test Lineage Tracker
def test_lineage_tracker(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test BenchmarkLineageTracker generating configuration hashes and provenance."""
    config_dict = {"test_key": "test_value"}
    lineage = BenchmarkLineageTracker.create_lineage(
        dataset_artifact=mock_dataset_artifact,
        config_dict=config_dict,
    )

    assert isinstance(lineage, BenchmarkLineage)
    assert lineage.dataset_artifact_id == mock_dataset_artifact.artifact_id
    assert len(lineage.configuration_hash) == 64


# 8. Test Reporting Generator
def test_report_generator(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test BenchmarkReportGenerator producing independent BenchmarkReport objects."""
    runner = BenchmarkRunner()
    profile = ProfileRegistry.get("teacher_evaluation")
    exec_results = runner.run_profile(mock_dataset_artifact, profile=profile)
    engine = BenchmarkScoringEngine()
    results, metrics, scores, summary = engine.evaluate(exec_results, profile=profile)

    report = BenchmarkReportGenerator.generate_report(
        results=results,
        metrics=metrics,
        scores=scores,
        summary=summary,
        profile=profile,
        dataset_artifact=mock_dataset_artifact,
        config_dict={},
    )

    assert isinstance(report, BenchmarkReport)
    assert report.overall_score == summary.overall_score
    assert len(report.metric_breakdown) == len(results)


# 9. Test Validator Engine
def test_benchmark_validator(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test BenchmarkValidator pre and post checks."""
    profile = ProfileRegistry.get("teacher_evaluation")
    report = BenchmarkValidator.validate_benchmark(
        dataset_artifact=mock_dataset_artifact, profile=profile
    )
    assert report.is_valid is True
    assert len(report.errors) == 0


# 10. Test Exporters
def test_benchmark_exporters(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test exporting BenchmarkArtifact to JSON, JSONL, Parquet, and Markdown."""
    pipeline = BenchmarkPipeline(profile_name="teacher_evaluation")
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact = pipeline.run(input_data=mock_dataset_artifact, output_dir=tmp_dir)

        descriptors = BenchmarkExporterManager.export_artifact(
            artifact=artifact,
            output_dir=tmp_dir,
            formats=["json", "jsonl", "parquet", "md"],
        )

        assert len(descriptors) == 4
        for d in descriptors:
            assert Path(d["path"]).exists()
            assert d["size_bytes"] > 0


# 11. Test End-to-End Pipeline Execution
def test_benchmark_pipeline_e2e(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test end-to-end flow: DatasetArtifact -> BenchmarkPipeline -> BenchmarkArtifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = BenchmarkPipeline(profile_name="dataset_quality")
        artifact = pipeline.run(
            input_data=mock_dataset_artifact,
            benchmark_id="bm_e2e_001",
            output_dir=tmp_dir,
        )

        assert isinstance(artifact, BenchmarkArtifact)
        assert artifact.artifact_id == "bm_e2e_001"
        assert artifact.scores.overall_score >= 0.0
        assert artifact.summary.passed_cases > 0
        assert len(artifact.export_files) > 0


# 12. Test CLI Commands
def test_benchmark_cli(mock_dataset_artifact: DatasetArtifact) -> None:
    """Test Typer CLI subcommands tsml benchmark run, report, validate, export, list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save mock dataset to disk
        ds_file = Path(tmp_dir) / "dataset.json"
        ds_file.write_text(mock_dataset_artifact.model_dump_json(), encoding="utf-8")

        # Test run
        res = cli_runner.invoke(
            cli_app,
            ["benchmark", "run", "--dataset", str(ds_file), "--output-dir", tmp_dir],
        )
        assert res.exit_code == 0
        assert "Benchmark Execution Completed Successfully!" in res.stdout

        # Test list
        res_list = cli_runner.invoke(cli_app, ["benchmark", "list"])
        assert res_list.exit_code == 0
        assert "teacher_evaluation" in res_list.stdout

        # Test report
        bm_file = Path(tmp_dir) / "coaching_benchmark_v1" / "coaching_benchmark_v1_benchmark.json"

        res_rpt = cli_runner.invoke(cli_app, ["benchmark", "report", str(bm_file)])
        assert res_rpt.exit_code == 0
        assert "Overall Score:" in res_rpt.stdout

        # Test validate
        res_val = cli_runner.invoke(cli_app, ["benchmark", "validate", str(bm_file)])
        assert res_val.exit_code == 0
        assert "is valid" in res_val.stdout

        # Test export
        res_exp = cli_runner.invoke(
            cli_app, ["benchmark", "export", str(bm_file), "--output-dir", tmp_dir]
        )
        assert res_exp.exit_code == 0
        assert "Successfully exported" in res_exp.stdout
