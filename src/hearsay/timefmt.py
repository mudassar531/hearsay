"""Timestamp formatting shared by sectioning and rendering."""


def format_timestamp(seconds: float) -> str:
    """Format seconds as zero-padded HH:MM:SS (floor; never negative)."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
