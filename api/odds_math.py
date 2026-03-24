# Shared American odds → implied probability (used by fair_value, parlay_engine).


def american_to_implied(american_odds):
    """Convert American odds (e.g. -140, +120) to implied probability [0, 1]."""
    if american_odds is None:
        return None
    try:
        odds = float(american_odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)
