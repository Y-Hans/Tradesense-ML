"""Market context schemas for TradeSense ML."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MarketRegime(str, Enum):
    """Market structural regime."""

    BULLISH_TREND = "BULLISH_TREND"
    BEARISH_TREND = "BEARISH_TREND"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY_BREAKOUT = "HIGH_VOLATILITY_BREAKOUT"
    BEARISH_BREAKDOWN = "BEARISH_BREAKDOWN"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY_CONSOLIDATION = "LOW_VOLATILITY_CONSOLIDATION"
    NEWS_DRIVEN_SPIKE = "NEWS_DRIVEN_SPIKE"
    FLASH_CRASH = "FLASH_CRASH"
    RECOVERY = "RECOVERY"


class VolatilityLevel(str, Enum):
    """Categorical volatility status."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TechnicalIndicators(BaseModel):
    """Snapshot of technical indicator values."""

    model_config = ConfigDict(frozen=True)

    rsi_14: float | None = Field(default=None, ge=0, le=100, description="14-period RSI")
    macd_histogram: float | None = Field(default=None, description="MACD Histogram")
    vwap_distance_pct: float | None = Field(
        default=None, description="Percentage distance from VWAP"
    )
    atr_14: float | None = Field(default=None, ge=0, description="14-period ATR")
    sma_50_above_200: bool | None = Field(default=None, description="Golden cross alignment")


class MarketContext(BaseModel):
    """Market condition context during trade entry/execution."""

    model_config = ConfigDict(frozen=True)

    context_id: str = Field(..., description="Unique market context snapshot ID")
    symbol: str = Field(..., description="Asset symbol")
    regime: MarketRegime = Field(..., description="Market regime classification")
    volatility: VolatilityLevel = Field(..., description="Volatility level")
    support_levels: list[float] = Field(
        default_factory=list, description="Key support price levels"
    )
    resistance_levels: list[float] = Field(
        default_factory=list, description="Key resistance price levels"
    )
    indicators: TechnicalIndicators = Field(
        default_factory=TechnicalIndicators, description="Technical indicator snapshot"
    )
    news_sentiment_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="News sentiment (-1 to +1)"
    )
    overall_trend_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Trend strength score (-1 to +1)"
    )
    context_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional market metadata"
    )
