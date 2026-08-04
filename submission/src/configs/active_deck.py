import os


def _detect_deck_from_csv() -> str:
    paths = ["deck.csv", "/kaggle_simulations/agent/deck.csv"]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "648" in content or "646" in content:
                        return "GRIMMSNARL"
                    if "431" in content or "150" in content or "151" in content:
                        return "MEWTWO"
                    if "400" in content:
                        return "LILLIE"
            except Exception:
                pass
    return "GRIMMSNARL"


ACTIVE_DECK = os.getenv("ACTIVE_DECK", _detect_deck_from_csv()).upper()


def get_active_deck_name() -> str:
    """Returns the current active deck name (e.g. 'MEWTWO', 'LILLIE', 'GRIMMSNARL')."""
    return ACTIVE_DECK


def set_active_deck(deck_name: str) -> None:
    """Programmatically updates the active deck setting."""
    global ACTIVE_DECK
    ACTIVE_DECK = deck_name.upper()

