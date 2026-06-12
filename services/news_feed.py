"""
services/news_feed.py
=====================
World financial headlines from free public RSS feeds — no API key required.
Each headline is tagged with an impact level (haute/normale) and the core
assets it likely concerns (GOLD / WTI / SPX / BTC), so the UI can filter and
the macro agent receives only relevant, pre-truncated, untrusted-as-data text.

Security notes:
- HTTPS feeds only.
- Titles are HTML-stripped, unescaped, truncated to 200 chars.
- Headlines are passed to the LLM explicitly framed as data, never instructions.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("news_feed")

FEEDS: tuple[tuple[str, str], ...] = (
    ("CNBC Markets", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Investing FR", "https://fr.investing.com/rss/news.rss"),
)

HIGH_IMPACT_KEYWORDS = (
    "fed", "fomc", "powell", "cpi", "ppi", "inflation", "rate cut", "rate hike",
    "taux", "bce", "ecb", "opec", "opep", "production cut", "nfp", "payroll",
    "jobs report", "gdp", "pib", "recession", "récession", "war", "guerre",
    "sanction", "tariff", "droits de douane", "default", "crash", "halving",
    "etf approval", "stimulus", "quantitative",
)

ASSET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "GOLD": ("gold", " or ", "bullion", "précieux", "xau"),
    "WTI": ("oil", "crude", "brent", "wti", "opec", "opep", "pétrole", "energy"),
    "SPX": ("s&p", "sp500", "wall street", "stocks", "equities", "nasdaq", "dow",
            "actions", "bourse", "earnings"),
    "BTC": ("bitcoin", "btc", "crypto", "ethereum", "blockchain", "etf"),
}

_TAG_RE = re.compile(r"<[^>]+>")

# Lexicon-based sentiment (transparent heuristic, NOT a deep model):
POSITIVE_WORDS = (
    "surge", "rally", "rallies", "soar", "record high", "beats", "beat ",
    "gains", "jumps", "rebound", "bullish", "optimism", "upgrade", "growth",
    "hausse", "progresse", "rebond", "record", "dépasse",
)
NEGATIVE_WORDS = (
    "crash", "plunge", "tumble", "slump", "fears", "fear ", "warning", "war",
    "guerre", "recession", "récession", "misses", "miss ", "falls", "drop",
    "sell-off", "selloff", "bearish", "downgrade", "default", "sanction",
    "baisse", "chute", "craint", "effondre", "tensions",
)


def sentiment_score(title: str, impact: str) -> int:
    """-10 (très négatif) .. +10 (très positif). Amplified for high-impact news."""
    lower = f" {title.lower()} "
    pos = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in lower)
    raw = 3 * (pos - neg)
    if impact == "haute":
        raw = int(raw * 1.5)
    return max(-10, min(10, raw))


@dataclass(frozen=True)
class Headline:
    source: str
    title: str
    link: str
    published: Optional[datetime]
    impact: str                       # 'haute' | 'normale'
    assets: tuple[str, ...] = field(default=())
    sentiment: int = 0                # -10 .. +10 (heuristique lexicale)

    @property
    def age_label(self) -> str:
        if self.published is None:
            return "—"
        delta = datetime.now(timezone.utc) - self.published
        mins = int(delta.total_seconds() // 60)
        if mins < 60:
            return f"il y a {mins} min"
        if mins < 60 * 24:
            return f"il y a {mins // 60} h"
        return f"il y a {mins // (60 * 24)} j"


def _clean_title(raw: str) -> str:
    return html.unescape(_TAG_RE.sub("", raw or "")).strip()[:200]


def _parse_time(entry) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        tm = getattr(entry, attr, None) or entry.get(attr)
        if tm:
            try:
                return datetime(*tm[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def _classify(title: str) -> tuple[str, tuple[str, ...]]:
    lower = f" {title.lower()} "
    impact = "haute" if any(k in lower for k in HIGH_IMPACT_KEYWORDS) else "normale"
    assets = tuple(a for a, kws in ASSET_KEYWORDS.items()
                   if any(k in lower for k in kws))
    return impact, assets


def _fetch_one_feed(source: str, url: str):
    """Download one feed with a HARD timeout, then parse from bytes.
    feedparser's own fetching has no reliable timeout — never let one slow
    vendor block the UI."""
    import feedparser
    import httpx
    try:
        r = httpx.get(url, timeout=4.0, follow_redirects=True,
                      headers={"User-Agent": "quant-terminal/1.0"})
        r.raise_for_status()
        return source, feedparser.parse(r.content)
    except Exception as exc:  # noqa: BLE001 — network boundary
        logger.warning("Feed failed (%s): %s", source, type(exc).__name__)
        return source, None


def fetch_headlines(limit: int = 50, per_feed: int = 15) -> list[Headline]:
    """Fetch ALL feeds IN PARALLEL (worst case ≈4s total instead of 20s+),
    clean, classify, score sentiment, dedupe and sort (newest first)."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        import feedparser  # noqa: F401 — fail fast if missing
    except ImportError:
        logger.warning("feedparser not installed — news feed disabled.")
        return []

    with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
        parsed = list(pool.map(lambda sf: _fetch_one_feed(*sf),
                               [(s, u) for s, u in FEEDS
                                if u.lower().startswith("https://")]))

    items: list[Headline] = []
    seen: set[str] = set()
    for source, feed in parsed:
        if feed is None:
            continue
        for entry in (feed.entries or [])[:per_feed]:
            title = _clean_title(entry.get("title", ""))
            if len(title) < 12:
                continue
            key = title.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            impact, assets = _classify(title)
            items.append(Headline(
                source=source,
                title=title,
                link=str(entry.get("link", ""))[:300],
                published=_parse_time(entry),
                impact=impact,
                assets=assets,
                sentiment=sentiment_score(title, impact),
            ))

    items.sort(key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
               reverse=True)
    return items[:limit]


def headlines_for_agents(headlines: list[Headline], max_items: int = 12) -> list[str]:
    """Compact, pre-truncated strings for the macro agent's context."""
    high = [h for h in headlines if h.impact == "haute"]
    rest = [h for h in headlines if h.impact == "normale"]
    chosen = (high + rest)[:max_items]
    return [f"[{h.impact.upper()}] {h.title} ({h.source})" for h in chosen]
