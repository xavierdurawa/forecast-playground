"""ForecastPlayground — a time-masked retrieval harness for AI forecasting.

Give a model tools that, as of a frozen date, return only information that existed
before that date — so it can be trained or evaluated on questions whose answers are
now known, without lookahead leakage.
"""

from .cache import ResultCache
from .clock import Clock, LookaheadError
from .schema import function_to_tool_def, tools_to_openai_schema
from .scoring import brier_score, log_score, mean_brier, mean_log
from .sources import (
    AsOfGuarantee,
    Document,
    PageviewsSource,
    PolymarketSource,
    ResolvedMarket,
    Source,
    WaybackSource,
    WikipediaSource,
    fetch_resolved_markets,
    market_prob_at,
    select_uncertain,
)
from .toolkit import Toolkit, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Clock",
    "LookaheadError",
    "AsOfGuarantee",
    "Document",
    "Source",
    "WikipediaSource",
    "PageviewsSource",
    "WaybackSource",
    "PolymarketSource",
    "ResolvedMarket",
    "fetch_resolved_markets",
    "market_prob_at",
    "select_uncertain",
    "Toolkit",
    "ToolCall",
    "ResultCache",
    "brier_score",
    "log_score",
    "mean_brier",
    "mean_log",
    "function_to_tool_def",
    "tools_to_openai_schema",
    "__version__",
]
