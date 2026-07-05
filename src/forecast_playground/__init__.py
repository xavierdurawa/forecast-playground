"""ForecastPlayground — a time-masked retrieval harness for AI forecasting.

Give a model tools that, as of a frozen date, return only information that existed
before that date — so it can be trained or evaluated on questions whose answers are
now known, without lookahead leakage.
"""

from .agent import Forecast, run_forecast
from .cache import ResultCache
from .calibration import CalibrationBin, CalibrationReport, calibration_report
from .clock import Clock, LookaheadError
from .drivers import AnthropicDriver, Driver, OpenAIDriver
from .env import load_env
from .judge import Judgment, brier_from_judgment, judge_forecast, parse_judgment
from .leakage import (
    is_leak_safe,
    min_safe_resolution,
    register_cutoff,
    training_cutoff,
)
from .scaffold import NAIVE, SUPERFORECASTER, Scaffold
from .schema import function_to_tool_def, tools_to_openai_schema
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
    GEO_SOURCES,
    AsOfGuarantee,
    CurrentEventsSource,
    Document,
    FREDSource,
    GDELTNewsSource,
    NOAASource,
    PageviewsSource,
    PolymarketSource,
    ResolvedMarket,
    ResolvedQuestion,
    Source,
    WaybackSource,
    WikipediaSource,
    fetch_forecastbench_questions,
    fetch_resolved_markets,
    market_prob_at,
    select_uncertain,
)
from .toolkit import Toolkit, ToolCall

__version__ = "0.1.0"

# The public API, grouped by role. Everything here is intended for consumers and
# kept stable; anything not listed (submodule internals) may change without notice.
__all__ = [
    "__version__",
    # --- Core: the leak-free foundation ---
    "Clock",              # the no-lookahead chokepoint
    "LookaheadError",
    "Source",             # protocol to implement your own time-masked tool
    "Document",           # what a Source returns (carries the guarded timestamp)
    "AsOfGuarantee",      # HARD / SOFT / NONE
    "Toolkit",            # binds sources + Clock into model-callable tools
    "ToolCall",           # per-call trace record
    "ResultCache",        # optional on-disk cache of tool results
    # --- Sources (all free, keyless) ---
    "WikipediaSource",
    "PageviewsSource",
    "WaybackSource",
    "PolymarketSource",
    "CurrentEventsSource",
    "GDELTNewsSource",
    "FREDSource",
    "NOAASource",
    # --- Datasets / question selection (Polymarket-backed) ---
    "ResolvedMarket",
    "fetch_resolved_markets",
    "market_prob_at",
    "select_uncertain",
    # --- Datasets: ForecastBench (Metaculus/INFER/ACLED + markets) ---
    "ResolvedQuestion",
    "fetch_forecastbench_questions",
    "GEO_SOURCES",
    # --- Driving a model to a forecast ---
    "run_forecast",
    "Forecast",
    "Driver",             # bring any provider
    "OpenAIDriver",       # OpenAI-compatible (OpenAI, vLLM, Ollama, ...)
    "AnthropicDriver",    # native Anthropic / Bedrock
    "Scaffold",           # swappable instructions/methodology
    "NAIVE",
    "SUPERFORECASTER",
    # --- Scoring, aggregation, calibration ---
    "brier_score",
    "log_score",
    "mean_brier",
    "mean_log",
    "trimmed_mean",       # ensemble seam: combine N sampled probabilities
    "extremize",
    "aggregate",
    "calibration_report",
    "CalibrationReport",
    "CalibrationBin",
    # --- LLM-as-judge scorer (non-deterministic; behind the [judge] extra) ---
    "judge_forecast",
    "parse_judgment",
    "brier_from_judgment",
    "Judgment",
    # --- Parametric leakage guard (training-cutoff filter) ---
    "is_leak_safe",
    "training_cutoff",
    "min_safe_resolution",
    "register_cutoff",
    # --- Tool-schema helpers ---
    "function_to_tool_def",
    "tools_to_openai_schema",
    # --- Local config ---
    "load_env",
]
