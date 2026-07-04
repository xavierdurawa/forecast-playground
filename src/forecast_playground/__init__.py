"""ForecastPlayground — a time-masked retrieval harness for AI forecasting.

Give a model tools that, as of a frozen date, return only information that existed
before that date — so it can be trained or evaluated on questions whose answers are
now known, without lookahead leakage.
"""

from .agent import Forecast, run_forecast
from .cache import ResultCache
from .clock import Clock, LookaheadError
from .drivers import AnthropicDriver, Driver, OpenAIDriver
from .scaffold import NAIVE, SUPERFORECASTER, Scaffold
from .schema import function_to_tool_def, tools_to_openai_schema
from .calibration import CalibrationBin, CalibrationReport, calibration_report
from .scoring import (
    aggregate,
    brier_score,
    extremize,
    log_score,
    mean_brier,
    mean_log,
    trimmed_mean,
)
from .sources import (
    AsOfGuarantee,
    CurrentEventsSource,
    Document,
    GDELTNewsSource,
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
    "CurrentEventsSource",
    "GDELTNewsSource",
    "ResolvedMarket",
    "fetch_resolved_markets",
    "market_prob_at",
    "select_uncertain",
    "Toolkit",
    "ToolCall",
    "ResultCache",
    "run_forecast",
    "Forecast",
    "Driver",
    "OpenAIDriver",
    "AnthropicDriver",
    "Scaffold",
    "NAIVE",
    "SUPERFORECASTER",
    "brier_score",
    "log_score",
    "mean_brier",
    "mean_log",
    "trimmed_mean",
    "extremize",
    "aggregate",
    "calibration_report",
    "CalibrationReport",
    "CalibrationBin",
    "function_to_tool_def",
    "tools_to_openai_schema",
    "__version__",
]
