"""Risk, discipline, and reason code schemas."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ReasonCodeCategory(str, Enum):
    """Broad categories for trading reason codes."""

    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    DISCIPLINE = "DISCIPLINE"
    EXECUTION = "EXECUTION"
    PSYCHOLOGY = "PSYCHOLOGY"
    STRATEGY = "STRATEGY"


class StandardReasonCode(str, Enum):
    """Standardized TradeSense reason codes."""

    # Risk Management
    R_NO_STOP_LOSS = "R_NO_STOP_LOSS"
    R_WIDE_STOP_LOSS = "R_WIDE_STOP_LOSS"
    R_EXCESSIVE_POSITION_SIZE = "R_EXCESSIVE_POSITION_SIZE"
    R_POOR_RISK_REWARD = "R_POOR_RISK_REWARD"
    R_OVERLEVERAGED = "R_OVERLEVERAGED"

    # Discipline & Psychology
    D_CHASE_ENTRY_FOMO = "D_CHASE_ENTRY_FOMO"
    D_REVENGE_TRADE = "D_REVENGE_TRADE"
    D_OVERTRADING = "D_OVERTRADING"
    D_PLAN_DEVIATION = "D_PLAN_DEVIATION"
    D_EARLY_EXIT_FEAR = "D_EARLY_EXIT_FEAR"

    # Positive Execution
    P_EXCELLENT_RR = "P_EXCELLENT_RR"
    P_PERFECT_PLAN_EXECUTION = "P_PERFECT_PLAN_EXECUTION"
    P_DISCIPLINED_STOP = "P_DISCIPLINED_STOP"


class ReasonCodeDetail(BaseModel):
    """Individual reason code annotation."""

    model_config = ConfigDict(frozen=True)

    code: StandardReasonCode | str = Field(..., description="Standard or custom reason code")
    category: ReasonCodeCategory = Field(..., description="Reason code category")
    explanation: str = Field(..., description="Contextual explanation for why code was assigned")
    severity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Severity rating (0=minor, 1=critical)"
    )


class RiskEvaluation(BaseModel):
    """Risk compliance evaluation model."""

    model_config = ConfigDict(frozen=True)

    risk_score: float = Field(..., ge=0.0, le=10.0, description="Overall risk score (0 to 10)")
    risk_reward_ratio: float | None = Field(
        default=None, ge=0.0, description="Calculated R:R ratio"
    )
    position_size_compliant: bool = Field(
        ..., description="Whether position sizing adheres to risk rules"
    )
    stop_loss_defined: bool = Field(..., description="Whether stop loss was specified")
    max_drawdown_risk_pct: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Potential max loss % of account"
    )
    risk_summary: str = Field(..., description="Qualitative summary of risk assessment")
    reason_codes: list[ReasonCodeDetail] = Field(
        default_factory=list, description="Identified risk reason codes"
    )


class DisciplineEvaluation(BaseModel):
    """Trading discipline and behavioral evaluation model."""

    model_config = ConfigDict(frozen=True)

    discipline_score: float = Field(
        ..., ge=0.0, le=10.0, description="Overall discipline score (0 to 10)"
    )
    fomo_indicator: bool = Field(..., description="Flag indicating potential FOMO entry")
    revenge_trade_indicator: bool = Field(
        ..., description="Flag indicating potential revenge trading"
    )
    overtrading_indicator: bool = Field(
        ..., description="Flag indicating excessive trading frequency"
    )
    plan_adherence_score: float = Field(
        ..., ge=0.0, le=10.0, description="Plan adherence rating (0 to 10)"
    )
    discipline_summary: str = Field(..., description="Qualitative summary of discipline assessment")
    reason_codes: list[ReasonCodeDetail] = Field(
        default_factory=list, description="Identified discipline reason codes"
    )
