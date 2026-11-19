"""
ClimFill — Automated Climate Station Data Gap-Filler.
"""
__version__ = "1.1.0"
__author__ = "Aran Castro"
__license__ = "MIT"

from .engine import ClimFillEngine, GapDetector, GapFiller, infer_bounds
from .reanalysis import get_open_meteo_data, suggest_reanalysis_mapping

__all__ = [
    "ClimFillEngine",
    "GapDetector",
    "GapFiller",
    "infer_bounds",
    "get_open_meteo_data",
    "suggest_reanalysis_mapping",
]
