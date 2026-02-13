"""LLM Agent — Groq-powered Chain-of-Thought trader (Llama 3.3)."""

from __future__ import annotations

import json
from typing import Any

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL


# ── Prompt template ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a professional quant trader. You will be given current market "
    "indicators and portfolio state for a single ticker. Analyze the data and "
    "decide whether to BUY, SELL, or HOLD. You MUST respond with ONLY valid "
    'JSON in this exact format: {"action": "BUY|SELL|HOLD", "reasoning": "..."}'
)

_USER_TEMPLATE = """Current market state for {ticker}:
- Price: ${price:.2f}  |  Volume: {volume:,.0f}
- RSI(14): {rsi:.2f}
- MACD: {macd:.4f}  |  Signal: {macd_signal:.4f}  |  Histogram: {macd_hist:.4f}
- EMA(20): {ema20:.2f}  |  EMA(50): {ema50:.2f}

Portfolio state:
- Cash: ${cash:,.2f}
- Holdings value: ${holdings:,.2f}
- Unrealised PnL: ${unrealised_pnl:,.2f}

Analyze the RSI and MACD trends. Decide: BUY, SELL, or HOLD.
Respond with ONLY valid JSON: {{"action": "BUY|SELL|HOLD", "reasoning": "..."}}"""


# ── LLM Trader class ────────────────────────────────────────────────────────


class LLMTrader:
    """Groq-powered trader using Llama 3.3 70B via the Groq API.

    Usage
    -----
    >>> trader = LLMTrader()
    >>> result = trader.predict(state_dict)
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = GROQ_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        self._client = Groq(api_key=api_key)
        self._model = model

    def predict(self, state_dict: dict[str, Any]) -> dict[str, str]:
        prompt = self._build_prompt(state_dict)

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content or ""
            return self._parse_json(raw_text)

        except Exception as e:
            print(f"DEBUG: LLM API Error: {e}")
            return {
                "action": "HOLD",
                "reasoning": f"API error — defaulting to HOLD: {e}",
            }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_prompt(self, state: dict[str, Any]) -> str:
        return _USER_TEMPLATE.format(**state)

    @staticmethod
    def _parse_json(text: str) -> dict[str, str]:
        try:
            data = json.loads(text)
            action = str(data.get("action", "")).upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"
            return {
                "action": action,
                "reasoning": str(data.get("reasoning", "")),
            }
        except json.JSONDecodeError:
            print(f"DEBUG: Failed to parse JSON. Raw:\n{text}\n")
            return {
                "action": "HOLD",
                "reasoning": f"JSON parse error. Raw: {text[:100]}",
            }
