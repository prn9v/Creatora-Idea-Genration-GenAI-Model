"""
GenAI Content Idea Generator
Uses Google Gemini to turn scraped trends + brand profile into 15 content ideas.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai   # pip install google-generativeai

from scrapers.social_scraper import TrendItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brand profile dataclass  (mirrors your Prisma BrandProfile model)
# ---------------------------------------------------------------------------

@dataclass
class BrandProfile:
    niche: Optional[str] = None
    audience: Optional[str] = None
    style_summary: Optional[str] = None
    avg_sentence_length: Optional[float] = None
    common_phrases: Optional[list] = None
    emotional_tone: Optional[str] = None
    formality_score: Optional[float] = None
    humor_usage: bool = False
    storytelling_style: Optional[str] = None
    topic_preferences: Optional[list] = None
    vocabulary_complexity: Optional[str] = None
    tone: Optional[str] = None

    @classmethod
    def from_prisma_dict(cls, data: dict) -> "BrandProfile":
        """Convert a Prisma/JSON dict (camelCase) into this dataclass."""
        return cls(
            niche=data.get("niche"),
            audience=data.get("audience"),
            style_summary=data.get("styleSummary"),
            avg_sentence_length=data.get("avgSentenceLength"),
            common_phrases=data.get("commonPhrases") or [],
            emotional_tone=data.get("emotionalTone"),
            formality_score=data.get("formalityScore"),
            humor_usage=data.get("humorUsage", False),
            storytelling_style=data.get("storytellingStyle"),
            topic_preferences=data.get("topicPreferences") or [],
            vocabulary_complexity=data.get("vocabularyComplexity"),
            tone=data.get("tone"),
        )


# ---------------------------------------------------------------------------
# Content idea output model (mirrors your Prisma ContentIdea)
# ---------------------------------------------------------------------------

@dataclass
class ContentIdea:
    title: str
    hook: str
    description: str           # 60-100 words
    format: str                # Reel | Carousel | Talking-head | BTS | Trend hijack | Story/Poll
    angle: str
    cta: str
    platform: str
    source: str = "TRENDING"   # AI | TRENDING
    trend_source: Optional[str] = None   # Reddit | YouTube | Instagram | X | Facebook
    trend_title: Optional[str] = None
    is_favorite: bool = False


# ---------------------------------------------------------------------------
# Gemini-powered idea generator
# ---------------------------------------------------------------------------

class ContentIdeaGenerator:
    """
    Sends brand profile + scraped trends to Gemini 1.5 Pro and parses
    exactly 15 ContentIdea objects from the structured JSON response.
    """

    MODEL = "gemini-2.5-flash"

    # Allowed values for validation
    VALID_FORMATS = {
        "Reel", "Carousel", "Talking-head", "BTS",
        "Trend hijack", "Story/Poll",
    }
    VALID_PLATFORMS = {"Instagram", "YouTube", "TikTok", "LinkedIn", "X", "Facebook"}

    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.85,       # creative but not chaotic
                top_p=0.95,
                max_output_tokens=8192,
            ),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        brand: BrandProfile,
        trends: list[TrendItem],
        idea_count: int = 15,
    ) -> list[ContentIdea]:
        prompt = self._build_prompt(brand, trends, idea_count)
        logger.debug("Sending prompt to Gemini (%d chars)", len(prompt))

        response = self.model.generate_content(prompt)
        raw = response.text
        logger.debug("Raw Gemini response length: %d", len(raw))

        ideas = self._parse_response(raw)
        logger.info("Parsed %d content ideas from Gemini", len(ideas))
        return ideas[:idea_count]   # hard cap

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        brand: BrandProfile,
        trends: list[TrendItem],
        count: int,
    ) -> str:
        brand_block = f"""
## Creator Brand Profile
- Niche: {brand.niche or 'Not specified'}
- Target audience: {brand.audience or 'Not specified'}
- Tone: {brand.tone or 'Not specified'}
- Emotional tone: {brand.emotional_tone or 'Not specified'}
- Style summary: {brand.style_summary or 'Not specified'}
- Formality score (0=casual, 1=formal): {brand.formality_score or 'N/A'}
- Uses humour: {'Yes' if brand.humor_usage else 'No'}
- Storytelling style: {brand.storytelling_style or 'Not specified'}
- Vocabulary complexity: {brand.vocabulary_complexity or 'Not specified'}
- Avg sentence length (words): {brand.avg_sentence_length or 'N/A'}
- Common phrases: {', '.join(brand.common_phrases or []) or 'None'}
- Topic preferences: {', '.join(str(t) for t in (brand.topic_preferences or [])) or 'None'}
""".strip()

        # Take up to 20 trending items to keep prompt concise
        trend_lines = []
        for i, t in enumerate(trends[:20], 1):
            trend_lines.append(
                f"{i}. [{t.platform}] \"{t.title}\" (score: {t.score}) — {t.description[:120]}"
            )
        trends_block = "\n".join(trend_lines) if trend_lines else "No trending data available."

        return f"""
You are an expert social media content strategist. Your job is to generate exactly {count} high-quality, actionable content ideas for a creator, based on their brand profile and current trending content across platforms.

{brand_block}

## Current Trending Content (sorted by engagement)
{trends_block}

---

## Your Task
Generate EXACTLY {count} content ideas tailored to this creator's brand and inspired by the trends above.

### Rules
1. Each idea must match the creator's niche, tone, and audience.
2. Mix the content formats: Reel, Carousel, Talking-head, BTS, Trend hijack, Story/Poll.
3. Mix platforms: Instagram, YouTube, TikTok, LinkedIn, X, Facebook.
4. If an idea is inspired by a trend, set source="TRENDING" and fill trendSource + trendTitle.
5. For pure AI-generated ideas with no trend reference, set source="AI".
6. description must be 60–100 words explaining the idea or the trend's meaning.
7. hook must grab attention in one sentence (max 15 words).
8. cta must be a specific call-to-action (e.g. "Comment your answer below", "Save this for later").

### Output Format
Return ONLY a valid JSON array — no markdown, no explanation, no code fences. 

[
  {{
    "title": "string (max 80 chars)",
    "hook": "string (max 15 words)",
    "description": "string (60-100 words)",
    "format": "Reel | Carousel | Talking-head | BTS | Trend hijack | Story/Poll",
    "angle": "string — the unique creative angle or POV",
    "cta": "string — specific call-to-action",
    "platform": "Instagram | YouTube | TikTok | LinkedIn | X | Facebook",
    "source": "AI | TRENDING",
    "trendSource": "Reddit | YouTube | Instagram | X | Facebook | null",
    "trendTitle": "string | null"
  }}
]
""".strip()

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> list[ContentIdea]:
        # Strip any accidental markdown fences Gemini might add
        clean = re.sub(r"```(?:json)?", "", raw).strip()

        # Find the outermost JSON array
        start = clean.find("[")
        end = clean.rfind("]") + 1
        if start == -1 or end == 0:
            logger.error("No JSON array found in Gemini response")
            return []

        try:
            items: list[dict] = json.loads(clean[start:end])
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s", exc)
            return []

        ideas: list[ContentIdea] = []
        for item in items:
            try:
                fmt = item.get("format", "Reel")
                if fmt not in self.VALID_FORMATS:
                    fmt = "Reel"

                plat = item.get("platform", "Instagram")
                if plat not in self.VALID_PLATFORMS:
                    plat = "Instagram"

                ideas.append(
                    ContentIdea(
                        title=str(item.get("title", ""))[:80],
                        hook=str(item.get("hook", "")),
                        description=str(item.get("description", "")),
                        format=fmt,
                        angle=str(item.get("angle", "")),
                        cta=str(item.get("cta", "")),
                        platform=plat,
                        source=item.get("source", "AI"),
                        trend_source=item.get("trendSource"),
                        trend_title=item.get("trendTitle"),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed idea item: %s", exc)

        return ideas