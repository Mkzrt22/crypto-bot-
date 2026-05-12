"""
Claude API Sentiment Analyzer — uses Claude to analyze news headlines
with deep contextual understanding.

Instead of simple keyword matching, sends batches of headlines to Claude
for professional-grade sentiment analysis.

Cost: ~$0.01 per batch (10-20 headlines every 5 minutes)
      ~$3/month at current rates

Requires: ANTHROPIC_API_KEY in .env
"""
import logging
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
# Sonnet 4.5 gives a good balance of analysis quality vs cost for sentiment classification.
# For even lower cost (~5× cheaper), swap to Haiku 4.5: "claude-haiku-4-5-20251001"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"


class ClaudeSentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("ANTHROPIC_API_KEY", getattr(config, "ANTHROPIC_API_KEY", ""))
        self._last_analysis: datetime | None = None
        self._cache_minutes: int = 30  # Analyze every 30 min (économie API ×6)
        self._cached_result: dict = {}
        self._enabled = bool(self.api_key)

        if not self._enabled:
            logger.warning("Claude sentiment disabled — set ANTHROPIC_API_KEY in .env")
        else:
            logger.info("Claude sentiment analyzer enabled")

    def analyze_headlines(self, headlines: list) -> dict:
        """Send headlines to Claude for sentiment analysis.

        Returns:
            {
                "score": float (-1 to +1),
                "label": str,
                "analysis": str (brief reasoning),
                "per_headline": [{title, sentiment, impact}],
                "market_impact": str,
            }
        """
        if not self._enabled:
            return self._cached_result or {"score": 0, "label": "Neutral", "analysis": "API key not set"}

        # Check cache
        now = datetime.now()
        if self._last_analysis and (now - self._last_analysis).total_seconds() < self._cache_minutes * 60:
            return self._cached_result

        if not headlines:
            return {"score": 0, "label": "Neutral", "analysis": "No headlines"}

        # Dédupliquer et prendre les 8 plus récentes
        recent = headlines[-8:]
        headlines_text = "\n".join(
            f"{i+1}. [{h.get('source', '?')}] {h.get('title', '')}"
            for i, h in enumerate(recent)
        )

        # Skip si headlines identiques au dernier appel
        import hashlib
        headlines_hash = hashlib.md5(headlines_text.encode()).hexdigest()
        if hasattr(self, '_last_headlines_hash') and self._last_headlines_hash == headlines_hash:
            logger.debug("Claude sentiment: headlines inchangées, skip appel API")
            return self._cached_result
        self._last_headlines_hash = headlines_hash

        prompt = f"""Rate crypto market sentiment from these headlines.
Score: -1.0 (bearish) to +1.0 (bullish). JSON only, no markdown.

{headlines_text}

{{"score":<float>,"label":"<Bearish|Slightly Bearish|Neutral|Slightly Bullish|Bullish>","analysis":"<1 sentence>","market_impact":"<low|medium|high|critical>"}}"""

        try:
            resp = requests.post(
                CLAUDE_API_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 250,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning(f"Claude API error: {resp.status_code} — {resp.text[:200]}")
                return self._cached_result or {"score": 0, "label": "Neutral", "analysis": "API error"}

            data = resp.json()
            text = data.get("content", [{}])[0].get("text", "")

            # Parse JSON response
            # Strip any markdown backticks if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            # Validate
            score = float(result.get("score", 0))
            score = max(-1.0, min(1.0, score))
            result["score"] = score

            self._cached_result = result
            self._last_analysis = now

            logger.info(
                f"Claude sentiment: {score:+.2f} ({result.get('label', '?')}) "
                f"impact={result.get('market_impact', '?')} — "
                f"{result.get('analysis', '')[:100]}"
            )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Claude response parse error: {e}")
            return self._cached_result or {"score": 0, "label": "Neutral", "analysis": "Parse error"}
        except Exception as e:
            logger.warning(f"Claude sentiment error: {e}")
            return self._cached_result or {"score": 0, "label": "Neutral", "analysis": str(e)}

    def get_score(self) -> float:
        """Get latest sentiment score."""
        return self._cached_result.get("score", 0.0)

    def get_score_normalized(self) -> float:
        """Get score as 0-100 for ML features."""
        return (self.get_score() + 1) * 50

    def get_analysis(self) -> str:
        """Get latest reasoning."""
        return self._cached_result.get("analysis", "No analysis yet")

    def get_market_impact(self) -> str:
        """Get market impact level."""
        return self._cached_result.get("market_impact", "low")

    def get_full_result(self) -> dict:
        """Get full analysis result."""
        return self._cached_result

    @property
    def is_enabled(self):
        return self._enabled
