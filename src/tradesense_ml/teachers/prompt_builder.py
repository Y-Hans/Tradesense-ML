"""Prompt Builder module for extracting and structuring scenario contexts for Teacher inference."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tradesense_ml.domain.schemas.coaching import CoachRequest


class PromptContext(BaseModel):
    """Structured, provider-agnostic context payload extracted from a CoachRequest."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., description="Unique coaching request ID")
    user_id: str = Field(..., description="User ID requesting coaching")
    trade_record_text: str = Field(..., description="Formatted summary of trade parameters")
    market_context_text: str = Field(..., description="Formatted summary of market context")
    user_notes_text: str = Field(..., description="Formatted user rationale/notes")
    requested_aspects_text: str = Field(..., description="Comma-separated requested coaching aspects")
    prompt_version: str = Field(default="v1", description="Target prompt template version")
    extra_context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context attributes"
    )


class PromptBuilder:
    """Dedicated prompt context builder for converting CoachRequest domain models into PromptContext."""

    def build_context(self, request: CoachRequest, prompt_version: str = "v1") -> PromptContext:
        """Extract trade, market context, user notes, and requested aspects into a structured PromptContext."""
        trade = request.trade
        stop_loss_val = getattr(trade, "initial_stop_loss", getattr(trade, "stop_loss", None))
        take_profit_val = getattr(trade, "initial_take_profit", getattr(trade, "take_profit", None))
        pnl_val = getattr(trade, "pnl", getattr(trade, "realized_pnl", None))
        leverage_val = getattr(trade, "leverage", "N/A")

        trade_summary_lines = [
            f"Trade ID: {trade.trade_id}",
            f"Symbol: {trade.symbol}",
            f"Side: {trade.side.value if hasattr(trade.side, 'value') else trade.side}",
            f"Entry Price: {trade.entry_price}",
            f"Exit Price: {trade.exit_price if trade.exit_price is not None else 'N/A'}",
            f"Quantity: {trade.quantity}",
            f"Stop Loss: {stop_loss_val if stop_loss_val is not None else 'None'}",
            f"Take Profit: {take_profit_val if take_profit_val is not None else 'None'}",
            f"Realized PnL: {pnl_val if pnl_val is not None else 'N/A'}",
            f"Leverage: {leverage_val}x" if leverage_val != "N/A" else "Leverage: N/A",
            f"Entry Timestamp: {trade.entry_timestamp}",
            f"Exit Timestamp: {trade.exit_timestamp if trade.exit_timestamp is not None else 'N/A'}",
        ]
        trade_record_text = "\n".join(trade_summary_lines)

        market_ctx = request.market_context
        if market_ctx:
            regime_val = market_ctx.regime.value if hasattr(market_ctx.regime, 'value') else market_ctx.regime
            vol_val = market_ctx.volatility.value if hasattr(market_ctx.volatility, 'value') else market_ctx.volatility
            ctx_id = getattr(market_ctx, "context_id", getattr(market_ctx, "scenario_id", "N/A"))
            market_lines = [
                f"Context ID: {ctx_id}",
                f"Market Symbol: {market_ctx.symbol}",
                f"Market Regime: {regime_val}",
                f"Volatility Level: {vol_val}",
            ]
            if hasattr(market_ctx, "trend_direction") and market_ctx.trend_direction:
                market_lines.append(f"Trend Direction: {market_ctx.trend_direction}")
            if market_ctx.indicators:
                ind = market_ctx.indicators
                market_lines.extend([
                    f"RSI (14): {ind.rsi_14 if ind.rsi_14 is not None else 'N/A'}",
                    f"ATR: {ind.atr_14 if hasattr(ind, 'atr_14') and ind.atr_14 is not None else 'N/A'}",
                    f"MACD Hist: {ind.macd_histogram if ind.macd_histogram is not None else 'N/A'}",
                ])
            market_context_text = "\n".join(market_lines)

        else:
            market_context_text = "No detailed market context provided."

        user_notes_text = request.user_notes if request.user_notes else "No notes provided by trader."
        requested_aspects_text = ", ".join(request.requested_aspects) if request.requested_aspects else "general"

        return PromptContext(
            request_id=request.request_id,
            user_id=request.user_id,
            trade_record_text=trade_record_text,
            market_context_text=market_context_text,
            user_notes_text=user_notes_text,
            requested_aspects_text=requested_aspects_text,
            prompt_version=prompt_version,
            extra_context={
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
            },
        )
