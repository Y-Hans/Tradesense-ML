"""Comprehensive tests for TradeSense ML Fine-Tuning Pipeline architecture."""

from pathlib import Path

from typer.testing import CliRunner

from tradesense_ml.cli.finetune_cmd import _create_mock_distillation_artifact, app
from tradesense_ml.domain.schemas.finetuning import (
    ModelArtifact,
    ModelConfiguration,
    ModelPackage,
    TrainingBackendConfiguration,
    TrainingConfiguration,
)
from tradesense_ml.finetuning.backends import (
    BackendRegistry,
    MockBackend,
)
from tradesense_ml.finetuning.checkpoint import CheckpointManager
from tradesense_ml.finetuning.evaluation import FineTuningEvaluationEngine
from tradesense_ml.finetuning.exporters import ModelExporter
from tradesense_ml.finetuning.lineage import FineTuningLineageTracker
from tradesense_ml.finetuning.pipeline import FineTuningPipeline
from tradesense_ml.finetuning.runner import FineTuningRunner
from tradesense_ml.finetuning.session import TrainingSession
from tradesense_ml.finetuning.statistics import FineTuningStatisticsGenerator
from tradesense_ml.finetuning.strategies import (
    CurriculumTrainingStrategy,
    DPOTrainingStrategy,
    HybridTrainingStrategy,
    ORPOTrainingStrategy,
    SFTTrainingStrategy,
    TrainingStrategyRegistry,
)
from tradesense_ml.finetuning.validation import FineTuningValidator

cli_runner = CliRunner()


def test_domain_models_schema(tmp_path: Path) -> None:
    """Test domain models schema instantiation and serialization."""
    model_cfg = ModelConfiguration(
        base_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        learning_rate=0.0001,
        num_epochs=2,
    )
    backend_cfg = TrainingBackendConfiguration(backend_name="mock")
    config = TrainingConfiguration(
        run_name="test-run",
        strategy_name="SFTTrainingStrategy",
        backend_config=backend_cfg,
        model_config_params=model_cfg,
    )
    assert config.run_name == "test-run"
    assert config.model_config_params.learning_rate == 0.0001

    dumped = config.model_dump(mode="json")
    reloaded = TrainingConfiguration.model_validate(dumped)
    assert reloaded.run_name == config.run_name


def test_backend_registry_and_backends(tmp_path: Path) -> None:
    """Test backend registry discovery and execution across all pluggable backends."""
    dist_artifact = _create_mock_distillation_artifact("dist_test")
    config = TrainingConfiguration(
        run_name="test-backend",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )

    available = BackendRegistry.list_available()
    assert "mock" in available
    assert "unsloth" in available
    assert "axolotl" in available
    assert "huggingface" in available
    assert "trl" in available

    for backend_name in ["mock", "unsloth", "axolotl", "huggingface", "trl"]:
        backend = BackendRegistry.get(backend_name)
        assert backend.name == backend_name
        result = backend.train(
            distillation_artifact=dist_artifact,
            config=config,
            output_dir=str(tmp_path / backend_name),
        )
        assert result.status == "success"
        assert Path(result.model_weights_dir).exists()


def test_strategy_registry_and_execution(tmp_path: Path) -> None:
    """Test training strategy registry and execution patterns."""
    dist_artifact = _create_mock_distillation_artifact("dist_strategy")
    backend = MockBackend()
    config = TrainingConfiguration(
        run_name="test-strategy",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )

    available_strategies = TrainingStrategyRegistry.list_available()
    assert "sfttrainingstrategy" in available_strategies
    assert "dpotrainingstrategy" in available_strategies
    assert "orpotrainingstrategy" in available_strategies
    assert "curriculumtrainingstrategy" in available_strategies
    assert "hybridtrainingstrategy" in available_strategies

    strategies = [
        SFTTrainingStrategy(),
        DPOTrainingStrategy(),
        ORPOTrainingStrategy(),
        CurriculumTrainingStrategy(),
        HybridTrainingStrategy(),
    ]

    for strat in strategies:
        res = strat.execute(
            distillation_artifact=dist_artifact,
            backend=backend,
            config=config,
            output_dir=str(tmp_path / strat.name),
        )
        assert res.status == "success"


def test_training_session_lifecycle(tmp_path: Path) -> None:
    """Test TrainingSession lifecycle state management and callback execution."""
    dist_artifact = _create_mock_distillation_artifact("dist_session")
    backend = MockBackend()
    strategy = SFTTrainingStrategy()
    config = TrainingConfiguration(
        run_name="test-session",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )

    session = TrainingSession(
        run_id="session-001",
        distillation_artifact=dist_artifact,
        config=config,
        backend=backend,
        strategy=strategy,
        output_dir=str(tmp_path / "session_dir"),
    )

    events_captured: list[str] = []

    def callback_fn(event: str, payload: dict) -> None:
        events_captured.append(event)

    session.register_callback(callback_fn)
    result = session.run()

    assert result.status == "success"
    assert "on_session_start" in events_captured
    assert "on_session_end" in events_captured
    assert session.execution_context.total_execution_seconds > 0.0


def test_checkpoint_manager(tmp_path: Path) -> None:
    """Test CheckpointManager registration, validation, best-checkpoint selection, and compilation."""
    mgr = CheckpointManager(output_dir=str(tmp_path))

    ckpt1 = mgr.register_checkpoint(
        step=100,
        epoch=1.0,
        loss=1.5,
        checkpoint_dir=str(tmp_path / "checkpoints" / "step-100"),
        eval_loss=1.6,
    )
    ckpt2 = mgr.register_checkpoint(
        step=200,
        epoch=2.0,
        loss=0.8,
        checkpoint_dir=str(tmp_path / "checkpoints" / "step-200"),
        eval_loss=0.9,
    )
    ckpt3 = mgr.register_checkpoint(
        step=300,
        epoch=3.0,
        loss=0.4,
        checkpoint_dir=str(tmp_path / "checkpoints" / "step-300"),
        eval_loss=0.5,
    )

    assert mgr.validate_checkpoint(str(tmp_path / "checkpoints" / "step-100"))

    res = mgr.compile_checkpoint_result([ckpt1, ckpt2, ckpt3])
    assert res.total_checkpoints_saved == 3
    assert res.best_checkpoint is not None
    assert res.best_checkpoint.checkpoint_id == "step-300"
    assert res.final_checkpoint.checkpoint_id == "step-300"


def test_evaluation_engine(tmp_path: Path) -> None:
    """Test FineTuningEvaluationEngine calculation of loss, perplexity, and benchmark scores."""
    backend = MockBackend()
    dist_artifact = _create_mock_distillation_artifact("dist_eval")
    config = TrainingConfiguration(
        run_name="test-eval",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )
    backend_res = backend.train(dist_artifact, config, str(tmp_path))

    eval_engine = FineTuningEvaluationEngine()
    eval_res = eval_engine.evaluate(backend_res)

    assert eval_res.eval_loss > 0.0
    assert eval_res.perplexity >= 1.0
    assert 0.0 <= eval_res.accuracy <= 1.0
    assert "trading_discipline_eval" in eval_res.benchmark_eval_scores


def test_statistics_lineage_and_validator(tmp_path: Path) -> None:
    """Test statistics generation, lineage tracking, and validator checks."""
    dist_artifact = _create_mock_distillation_artifact("dist_stats")
    backend = MockBackend()
    config = TrainingConfiguration(
        run_name="test-stats",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )
    backend_res = backend.train(dist_artifact, config, str(tmp_path))

    ckpt_mgr = CheckpointManager(output_dir=str(tmp_path))
    ckpt_res = ckpt_mgr.compile_checkpoint_result(backend_res.checkpoints)

    stats_gen = FineTuningStatisticsGenerator()
    tr_stats = stats_gen.generate_training_statistics(
        distillation_artifact=dist_artifact,
        backend_result=backend_res,
        model_config=config.model_config_params,
        checkpoint_result=ckpt_res,
    )
    assert tr_stats.total_steps > 0

    eval_engine = FineTuningEvaluationEngine()
    eval_res = eval_engine.evaluate(backend_res)
    model_stats = stats_gen.generate_model_statistics(tr_stats, eval_res)
    assert model_stats.memory_usage_mb > 0.0

    session = TrainingSession(
        run_id="run-lineage",
        distillation_artifact=dist_artifact,
        config=config,
        backend=backend,
        strategy=SFTTrainingStrategy(),
        output_dir=str(tmp_path),
    )
    lineage_tracker = FineTuningLineageTracker()
    lineage = lineage_tracker.generate_lineage(
        model_id="model-lineage",
        distillation_artifact=dist_artifact,
        config=config,
        backend_name="mock",
        strategy_name="SFTTrainingStrategy",
        execution_context=session.execution_context,
    )
    assert lineage.configuration_hash is not None

    validator = FineTuningValidator()
    compat_issues = validator.validate_distillation_artifact(dist_artifact)
    assert len(compat_issues) == 0

    cfg_issues = validator.validate_training_configuration(config)
    assert len(cfg_issues) == 0


def test_reporting_packaging_and_exporters(tmp_path: Path) -> None:
    """Test reporting, model packager (producing ModelPackage), and exporter execution."""
    pipeline = FineTuningPipeline()
    dist_artifact = _create_mock_distillation_artifact("dist_export")
    config = TrainingConfiguration(
        run_name="test-export",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )
    model_artifact = pipeline.run(
        input_data=dist_artifact, config=config, output_dir=str(tmp_path / "pipeline_run")
    )

    assert isinstance(model_artifact, ModelArtifact)
    assert isinstance(model_artifact.package, ModelPackage)
    assert Path(model_artifact.package.config_path).exists()
    assert Path(model_artifact.package.manifest_path).exists()

    exporter = ModelExporter()
    export_descs = exporter.export(
        artifact=model_artifact,
        output_dir=str(tmp_path / "exported_models"),
        export_formats=["directory", "json", "markdown", "manifest", "huggingface"],
    )
    assert len(export_descs) == 5


def test_finetuning_runner_retries(tmp_path: Path) -> None:
    """Test FineTuningRunner strategy resolution and execution."""
    runner = FineTuningRunner(max_retries=1)
    dist_artifact = _create_mock_distillation_artifact("dist_runner")
    config = TrainingConfiguration(
        run_name="test-runner",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
    )
    res, session = runner.run(
        run_id="run-001",
        distillation_artifact=dist_artifact,
        config=config,
        output_dir=str(tmp_path / "runner_out"),
    )
    assert res.status == "success"
    assert session.run_id == "run-001"


def test_finetuning_pipeline_e2e(tmp_path: Path) -> None:
    """End-to-End test for FineTuningPipeline consuming DistillationArtifact and producing ModelArtifact."""
    dist_artifact = _create_mock_distillation_artifact("dist_e2e")
    pipeline = FineTuningPipeline()
    config = TrainingConfiguration(
        run_name="e2e-coaching-model",
        strategy_name="SFTTrainingStrategy",
        backend_config=TrainingBackendConfiguration(backend_name="mock"),
        output_dir=str(tmp_path / "e2e_out"),
    )

    model_artifact = pipeline.run(
        input_data=dist_artifact, config=config, output_dir=str(tmp_path / "e2e_out")
    )

    assert model_artifact.artifact_id.startswith("model-ft-e2e-coaching-model")
    assert model_artifact.summary.final_loss > 0.0
    assert model_artifact.package.package_checksum is not None
    assert len(model_artifact.export_files) > 0


def test_cli_finetune_commands(tmp_path: Path) -> None:
    """Test Typer CLI `tsml finetune` commands."""
    # Test CLI run
    res_run = cli_runner.invoke(
        app,
        [
            "run",
            "--run-name",
            "cli-test-model",
            "--epochs",
            "1",
            "--output-dir",
            str(tmp_path / "cli_out"),
        ],
    )
    assert res_run.exit_code == 0
    assert "Fine-Tuning completed successfully" in res_run.output

    # Test CLI validate
    art_file = tmp_path / "cli_out" / "ft-cli-test-model" / "model_export" / "model_artifact.json"
    if art_file.exists():
        res_val = cli_runner.invoke(app, ["validate", str(art_file)])
        assert res_val.exit_code == 0
        assert "Validation PASSED" in res_val.output

    # Test CLI checkpoints
    res_ckpts = cli_runner.invoke(
        app, ["checkpoints", str(tmp_path / "cli_out" / "ft-cli-test-model")]
    )
    assert res_ckpts.exit_code == 0
