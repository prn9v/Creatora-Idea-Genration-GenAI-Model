"""
Content Ideas Pipeline
======================
Ties scraping + GenAI generation together into one clean function call.

Usage (from NestJS/TypeScript via HTTP, or directly in Python):

    from pipeline import run_pipeline, BrandProfileInput

    ideas = run_pipeline(brand_data={...})   # brand_data is your Prisma JSON dict
"""

import logging
import os
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv          # pip install python-dotenv

load_dotenv()  # reads .env at project root

from scrapers.social_scraper import SocialMediaOrchestrator, TrendItem
from genai.idea_generator import BrandProfile, ContentIdea, ContentIdeaGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword extractor — derives search keywords from brand profile
# ---------------------------------------------------------------------------

def extract_keywords(brand: BrandProfile) -> list[str]:
    """
    Build search keywords from the brand profile so scrapers know what to look for.
    Falls back to generic creator keywords if profile is sparse.
    """
    kws: set[str] = set()

    if brand.niche:
        kws.add(brand.niche)

    if brand.topic_preferences:
        for tp in brand.topic_preferences[:4]:
            kws.add(str(tp))

    if brand.common_phrases:
        # Only the shortest phrases tend to be good search terms
        for phrase in brand.common_phrases[:3]:
            if len(str(phrase).split()) <= 3:
                kws.add(str(phrase))

    if brand.audience:
        # e.g. "fitness beginners" → add "fitness"
        kws.add(brand.audience.split()[0] if brand.audience else "")

    # Always keep at least 2 keywords
    kws.discard("")
    if len(kws) < 2:
        kws.update(["content creation", "social media trends"])

    return list(kws)[:6]   # cap at 6 to stay within API rate limits


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    brand_data: dict[str, Any],
    idea_count: int = 15,
) -> list[dict[str, Any]]:
    """
    Full pipeline:
      1. Parse brand profile
      2. Derive search keywords
      3. Scrape all social platforms
      4. Generate 15 AI content ideas
      5. Return a list of plain dicts (ready to JSON-serialize back to NestJS)

    Args:
        brand_data:  The Prisma BrandProfile JSON dict (camelCase keys).
        idea_count:  How many ideas to generate (default 15).

    Returns:
        List of content idea dicts matching your ContentIdea Prisma shape.
    """

    # 1. Parse brand
    brand = BrandProfile.from_prisma_dict(brand_data)
    logger.info("Brand parsed — niche: '%s', audience: '%s'", brand.niche, brand.audience)

    # 2. Derive keywords
    keywords = extract_keywords(brand)
    logger.info("Search keywords: %s", keywords)

    # 3. Scrape
    orchestrator = SocialMediaOrchestrator()
    trends: list[TrendItem] = orchestrator.collect(keywords, per_platform_limit=8)
    logger.info("Total trend items collected: %d", len(trends))

    # 4. Generate ideas
    generator = ContentIdeaGenerator()
    ideas: list[ContentIdea] = generator.generate(brand, trends, idea_count=idea_count)

    # 5. Serialize to plain dicts (camelCase to match Prisma/TypeScript)
    def to_dict(idea: ContentIdea) -> dict:
        return {
            "title": idea.title,
            "hook": idea.hook,
            "description": idea.description,
            "format": idea.format,
            "angle": idea.angle,
            "cta": idea.cta,
            "platform": idea.platform,
            "source": idea.source,
            "trendSource": idea.trend_source,
            "trendTitle": idea.trend_title,
            "isFavorite": False,
        }

    return [to_dict(idea) for idea in ideas]


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Paste a sample brand profile here to test locally
    sample_brand = {
        "niche": "fitness and wellness",
        "audience": "women 25-35 who want to lose weight",
        "styleSummary": "Motivational, relatable, science-backed",
        "tone": "ENERGETIC",
        "emotionalTone": "Empowering",
        "formalityScore": 0.3,
        "humorUsage": True,
        "storytellingStyle": "Personal anecdotes with actionable tips",
        "vocabularyComplexity": "Simple",
        "avgSentenceLength": 12.5,
        "topicPreferences": ["weight loss", "home workouts", "nutrition", "mindset"],
        "commonPhrases": ["you've got this", "small steps", "sustainable habits"],
    }

    import json
    results = run_pipeline(sample_brand, idea_count=15)
    print(json.dumps(results, indent=2))