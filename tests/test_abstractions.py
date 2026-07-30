"""Unit tests for AssetManager, TeacherRouter, and Storage abstractions."""

import tempfile

from tradesense_ml.assets_manager.manager import AssetManager
from tradesense_ml.domain.schemas import (
    Side,
    TeacherRequest,
    Trade,
)
from tradesense_ml.storage.base import LocalDiskStorageBackend
from tradesense_ml.teachers.providers.openrouter import OpenRouterTeacherProvider
from tradesense_ml.teachers.router import TeacherRouter


def test_asset_manager_loading() -> None:
    """Test AssetManager loading prompts, rubrics, templates, and benchmarks."""
    manager = AssetManager()
    prompt = manager.get_prompt("system_coach", version="v1")
    assert "TradeSense Coach" in prompt

    rubric = manager.get_rubric("risk_discipline_rubric", version="v1")
    assert rubric.rubric_id == "risk_discipline_v1"
    assert len(rubric.criteria) == 4

    benchmark = manager.get_benchmark("coaching_benchmark_v1", version="v1")
    assert benchmark["benchmark_id"] == "coaching_benchmark_v1"

    template = manager.get_template("market_context_template")
    assert template["template_id"] == "market_context_v1"


def test_teacher_router_and_cost_estimation() -> None:
    """Test TeacherRouter registration and cost estimation."""
    provider = OpenRouterTeacherProvider(default_model="anthropic/claude-3.5-sonnet")
    router = TeacherRouter([provider])

    request = TeacherRequest(
        request_id="req_test_10",
        system_prompt="System instructions",
        user_prompt="Evaluate trade",
    )

    response = router.route(request, target_provider="openrouter")
    assert response.provider_metadata.provider_name == "openrouter"
    assert response.usage.prompt_tokens == 150
    assert response.usage.completion_tokens == 300
    assert response.usage.estimated_cost_usd > 0.0


def test_local_storage_backend() -> None:
    """Test LocalDiskStorageBackend save, load, exists, list_keys."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = LocalDiskStorageBackend[Trade](root_path=tmp_dir)

        trade = Trade(
            trade_id="t_store_1",
            user_id="u1",
            symbol="EUR/USD",
            side=Side.BUY,
            entry_price=1.0850,
            quantity=10000.0,
            entry_timestamp="2026-07-30T12:00:00Z",
        )

        storage.save("trade_001", trade)
        assert storage.exists("trade_001")
        assert not storage.exists("trade_999")

        loaded = storage.load("trade_001", Trade)
        assert loaded.trade_id == "t_store_1"
        assert loaded.symbol == "EUR/USD"

        keys = storage.list_keys()
        assert "trade_001" in keys
