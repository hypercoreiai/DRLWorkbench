# Plot/style configuration (V3)

from typing import Any, Dict, Optional

# Central place for matplotlib style defaults
STYLE_DEFAULTS: Dict[str, Any] = {
    "figure.figsize": (10, 6),
    "font.size": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
}


def apply_style(style_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Apply matplotlib style. Merges with STYLE_DEFAULTS.

    Args:
        style_dict: Optional overrides.
    """
    try:
        import matplotlib.pyplot as plt
        d = {**STYLE_DEFAULTS}
        if style_dict:
            d.update(style_dict)
        plt.rcParams.update(d)
    except ImportError:
        pass
