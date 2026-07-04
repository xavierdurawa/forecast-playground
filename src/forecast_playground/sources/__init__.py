"""Time-masked data source adapters."""

from .base import AsOfGuarantee, Document, Source
from .current_events import CurrentEventsSource
from .gdelt import GDELTNewsSource
from .pageviews import PageviewsSource
from .polymarket import (
    PolymarketSource,
    ResolvedMarket,
    fetch_resolved_markets,
    market_prob_at,
    select_uncertain,
)
from .wayback import WaybackSource
from .wikipedia import WikipediaSource

__all__ = [
    "AsOfGuarantee",
    "Document",
    "Source",
    "WikipediaSource",
    "PageviewsSource",
    "WaybackSource",
    "PolymarketSource",
    "CurrentEventsSource",
    "GDELTNewsSource",
    "ResolvedMarket",
    "fetch_resolved_markets",
    "market_prob_at",
    "select_uncertain",
]
