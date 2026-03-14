"""
Social Media Scraper — Multi-platform content trend collector
Supports: Reddit, YouTube, Instagram, Facebook, X (Twitter)
"""

import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
from typing import List
import requests
import httpx
import praw                          # pip install praw
from googleapiclient.discovery import build  # pip install google-api-python-client
from instagrapi import Client as InstaClient  # pip install instagrapi
import tweepy                        # pip install tweepy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model shared across all scrapers
# ---------------------------------------------------------------------------

@dataclass
class TrendItem:
    title: str
    description: str
    platform: str                    # Reddit | YouTube | Instagram | Facebook | X
    url: str = ""
    score: int = 0                   # upvotes / views / likes
    tags: list[str] = field(default_factory=list)
    scraped_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------
# class RedditScraper:
#     """
#     Uses Pushshift API instead of Reddit OAuth.
#     No API keys required.
#     """

#     BASE_URL = "https://api.pushshift.io/reddit/search/submission/"

#     def scrape(self, keywords: list[str], limit: int = 10) -> list["TrendItem"]:
#         results: List[TrendItem] = []

#         for kw in keywords:
#             try:
#                 params = {
#                     "q": kw,
#                     "subreddit": "startups",
#                     "sort": "desc",
#                     "sort_type": "score",
#                     "size": limit,
#                 }

#                 response = requests.get(self.BASE_URL, params=params, timeout=10)
#                 response.raise_for_status()
#                 data = response.json()

#                 for post in data.get("data", []):
#                     results.append(
#                         TrendItem(
#                             title=post.get("title", ""),
#                             description=(post.get("selftext") or "")[:300],
#                             platform="Reddit",
#                             url=f"https://reddit.com{post.get('permalink','')}",
#                             score=post.get("score", 0),
#                             tags=[post.get("subreddit", "startups")],
#                         )
#                     )

#             except Exception as exc:
#                 logger.warning("Pushshift Reddit scrape error for '%s': %s", kw, exc)

#         return results


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

class YouTubeScraper:
    """
    Uses the official YouTube Data API v3.
    Get an API key at: https://console.cloud.google.com
    """

    def __init__(self):
        self.yt = build(
            "youtube", "v3",
            developerKey=os.environ["YOUTUBE_API_KEY"],
            cache_discovery=False,
        )

    def scrape(self, keywords: list[str], limit: int = 10) -> list[TrendItem]:
        results: list[TrendItem] = []
        for kw in keywords:
            try:
                resp = (
                    self.yt.search()
                    .list(
                        q=kw,
                        part="snippet",
                        type="video",
                        order="viewCount",
                        publishedAfter="2024-01-01T00:00:00Z",
                        maxResults=limit,
                    )
                    .execute()
                )
                for item in resp.get("items", []):
                    snip = item["snippet"]
                    vid_id = item["id"]["videoId"]
                    results.append(
                        TrendItem(
                            title=snip["title"],
                            description=snip.get("description", "")[:300],
                            platform="YouTube",
                            url=f"https://youtube.com/watch?v={vid_id}",
                            tags=snip.get("tags", [])[:5],
                        )
                    )
            except Exception as exc:
                logger.warning("YouTube scrape error for '%s': %s", kw, exc)
        return results


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------

class InstagramScraper:
    """
    Uses instagrapi (unofficial, session-based).
    For production, use Instagram Graph API with a Business account.
    Credentials: IG_USERNAME / IG_PASSWORD in .env
    """

    def __init__(self):
        self.cl = InstaClient()
        self.cl.login(
            os.environ["IG_USERNAME"],
            os.environ["IG_PASSWORD"],
        )

    def scrape(self, keywords: list[str], limit: int = 10) -> list[TrendItem]:
        results: list[TrendItem] = []
        for kw in keywords:
            try:
                medias = self.cl.hashtag_medias_top(kw.lstrip("#"), amount=limit)
                for m in medias:
                    results.append(
                        TrendItem(
                            title=f"#{kw} post",
                            description=(m.caption_text or "")[:300],
                            platform="Instagram",
                            url=f"https://instagram.com/p/{m.code}/",
                            score=m.like_count,
                            tags=[kw],
                        )
                    )
            except Exception as exc:
                logger.warning("Instagram scrape error for '%s': %s", kw, exc)
        return results


# ---------------------------------------------------------------------------
# X (Twitter)
# ---------------------------------------------------------------------------

class XScraper:
    """
    Uses Twitter API v2 via Tweepy.
    Get bearer token at: https://developer.twitter.com
    """

    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=os.environ["X_BEARER_TOKEN"],
            wait_on_rate_limit=True,
        )

    def scrape(self, keywords: list[str], limit: int = 10) -> list[TrendItem]:
        results: list[TrendItem] = []
        for kw in keywords:
            try:
                resp = self.client.search_recent_tweets(
                    query=f"{kw} -is:retweet lang:en",
                    max_results=min(limit, 100),
                    tweet_fields=["public_metrics", "created_at"],
                )
                for tweet in resp.data or []:
                    results.append(
                        TrendItem(
                            title=tweet.text[:120],
                            description=tweet.text[:300],
                            platform="X",
                            url=f"https://twitter.com/i/web/status/{tweet.id}",
                            score=tweet.public_metrics.get("like_count", 0),
                            tags=[kw],
                        )
                    )
            except Exception as exc:
                logger.warning("X scrape error for '%s': %s", kw, exc)
        return results


# ---------------------------------------------------------------------------
# Facebook (Meta Graph API — requires a Page Access Token)
# ---------------------------------------------------------------------------

class FacebookScraper:
    """
    Uses Meta Graph API to search public page posts.
    Requires: FB_ACCESS_TOKEN (Page Access Token)
    Docs: https://developers.facebook.com/docs/graph-api
    """

    BASE_URL = "https://graph.facebook.com/v19.0"

    def __init__(self):
        self.token = os.environ["FB_ACCESS_TOKEN"]

    def scrape(self, keywords: list[str], limit: int = 10) -> list[TrendItem]:
        results: list[TrendItem] = []
        for kw in keywords:
            try:
                resp = httpx.get(
                    f"{self.BASE_URL}/search",
                    params={
                        "q": kw,
                        "type": "post",
                        "limit": limit,
                        "access_token": self.token,
                        "fields": "id,message,created_time,permalink_url",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                for post in resp.json().get("data", []):
                    results.append(
                        TrendItem(
                            title=(post.get("message", "") or "")[:120],
                            description=(post.get("message", "") or "")[:300],
                            platform="Facebook",
                            url=post.get("permalink_url", ""),
                            tags=[kw],
                        )
                    )
            except Exception as exc:
                logger.warning("Facebook scrape error for '%s': %s", kw, exc)
        return results


# ---------------------------------------------------------------------------
# Orchestrator — runs all scrapers and merges results
# ---------------------------------------------------------------------------

class SocialMediaOrchestrator:
    """
    Instantiates all available scrapers and merges their results.
    Any scraper whose env vars are missing is skipped gracefully.
    """

    def __init__(self):
        self.scrapers: list = []
        # self.scrapers.append(RedditScraper())
        # logger.info("✓ RedditScraper (Pushshift) loaded")
        self._try_add(YouTubeScraper, ["YOUTUBE_API_KEY"])
        self._try_add(InstagramScraper, ["IG_USERNAME", "IG_PASSWORD"])
        self._try_add(XScraper, ["X_BEARER_TOKEN"])
        self._try_add(FacebookScraper, ["FB_ACCESS_TOKEN"])

    def _try_add(self, cls, required_env: list[str]):
        if all(os.environ.get(k) for k in required_env):
            try:
                self.scrapers.append(cls())
                logger.info("✓ %s loaded", cls.__name__)
            except Exception as exc:
                logger.warning("✗ %s failed to init: %s", cls.__name__, exc)
        else:
            missing = [k for k in required_env if not os.environ.get(k)]
            logger.info("– %s skipped (missing: %s)", cls.__name__, missing)

    def collect(self, keywords: list[str], per_platform_limit: int = 8) -> list[TrendItem]:
        all_items: list[TrendItem] = []
        for scraper in self.scrapers:
            items = scraper.scrape(keywords, limit=per_platform_limit)
            all_items.extend(items)
            logger.info(
                "%s → %d items collected", scraper.__class__.__name__, len(items)
            )
        # Sort by score descending so the best content bubbles up
        all_items.sort(key=lambda x: x.score, reverse=True)
        return all_items