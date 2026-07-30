"""Trade schema definitions for TradeSense ML."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Side(str, Enum):
    """Trade direction."""

    BUY = "BUY"
    SELL = "SELL"
    LONG = "LONG"
    SHORT = "SHORT"


class TimeFrame(str, Enum):
    """Standard timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class TradeExecution(BaseModel):
    """Details of a single execution fill."""

    model_config = ConfigDict(frozen=True)

    execution_id: str = Field(..., description="Unique fill ID")
    price: float = Field(..., gt=0, description="Fill price")
    quantity: float = Field(..., gt=0, description="Filled quantity")
    timestamp: datetime = Field(..., description="Execution timestamp")
    fee: float = Field(default=0.0, ge=0, description="Execution fee")


class TradeOrder(BaseModel):
    """Order specification associated with a trade."""

    model_config = ConfigDict(frozen=True)

    order_id: str = Field(..., description="Unique order ID")
    side: Side = Field(..., description="Order direction")
    limit_price: float | None = Field(default=None, description="Limit price if applicable")
    stop_price: float | None = Field(default=None, description="Stop price if applicable")
    stop_loss: float | None = Field(default=None, description="Configured stop loss")
    take_profit: float | None = Field(default=None, description="Configured take profit")


class Trade(BaseModel):
    """Complete representation of a user trade."""

    model_config = ConfigDict(frozen=True)

    trade_id: str = Field(..., description="Unique trade ID")
    user_id: str = Field(..., description="User / Trader ID")
    symbol: str = Field(..., description="Asset symbol, e.g. AAPL, BTC/USD")
    side: Side = Field(..., description="Trade side")
    entry_price: float = Field(..., gt=0, description="Average entry price")
    exit_price: float | None = Field(default=None, description="Average exit price if closed")
    quantity: float = Field(..., gt=0, description="Position size / quantity")
    initial_stop_loss: float | None = Field(default=None, description="Initial stop loss price")
    initial_take_profit: float | None = Field(default=None, description="Initial take profit price")
    pnl: float | None = Field(default=None, description="Realized PnL")
    pnl_percentage: float | None = Field(default=None, description="Realized PnL percentage")
    timeframe: TimeFrame = Field(default=TimeFrame.M5, description="Primary timeframe")
    entry_timestamp: datetime = Field(..., description="Position entry timestamp")
    exit_timestamp: datetime | None = Field(default=None, description="Position exit timestamp")
    orders: list[TradeOrder] = Field(default_factory=list, description="Associated orders")
    executions: list[TradeExecution] = Field(
        default_factory=list, description="Associated executions"
    )
    tags: list[str] = Field(default_factory=list, description="Custom user/system tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata attributes"
    )
