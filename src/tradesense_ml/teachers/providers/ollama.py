"""Ollama Teacher provider interface stub."""

import json
from typing import Any

from tradesense_ml.domain.schemas.teacher import TeacherRequest
from tradesense_ml.teachers.base import BaseTeacherProvider


class OllamaTeacherProvider(BaseTeacherProvider):
    """Ollama local model provider implementation interface."""

    def __init__(self, default_model: str = "llama3:8b") -> None:
        super().__init__(
            provider_name="ollama",
            default_model=default_model,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )

    def _do_generate(self, request: TeacherRequest) -> tuple[str, dict[str, Any] | None, int, int]:
        parsed_dict = {
            "response_id": f"resp_{request.request_id}",
            "request_id": request.request_id,
            "headline": f"[Ollama:{self.default_model}] Coaching evaluation generated.",
            "overall_score": 8.0,
            "risk_evaluation": {
                "risk_score": 8.0,
                "risk_reward_ratio": 2.0,
                "position_size_compliant": True,
                "stop_loss_defined": True,
                "max_drawdown_risk_pct": 2.0,
                "risk_summary": "Risk parameters adhered to guidelines.",
                "reason_codes": [],
            },
            "discipline_evaluation": {
                "discipline_score": 8.0,
                "fomo_indicator": False,
                "revenge_trade_indicator": False,
                "overtrading_indicator": False,
                "plan_adherence_score": 8.0,
                "discipline_summary": "Followed trading plan.",
                "reason_codes": [],
            },
            "actionable_advice": ["Maintain strict risk limits."],
            "educational_note": "Consistency is key.",
        }
        return json.dumps(parsed_dict), parsed_dict, 170, 290
