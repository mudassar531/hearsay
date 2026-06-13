"""Timestamp formatting shared by sectioning and rendering."""

import math


def format_timestamp(seconds: float) -> str:
    """Format seconds as zero-padded HH:MM:SS (floor; never negative).

    Non-finite input (NaN/inf, which can slip through malformed metadata)
    formats as ``00:00:00`` rather than raising.
    """
    total = max(0, int(seconds)) if math.isfinite(seconds) else 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
