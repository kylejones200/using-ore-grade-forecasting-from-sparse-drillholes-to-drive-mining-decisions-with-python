"""
Shared Tufte-style plotting utilities for medium-repos.

Import pattern (already used by visualization files):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml


COLORS: dict[str, str] = {
    "black": "#2b2b2b",
    "darkgray": "#696969",
    "gray": "#a0a0a0",
    "lightgray": "#d3d3d3",
    "accent_red": "#c0392b",
    "accent_blue": "#2980b9",
    "accent_green": "#27ae60",
    "white": "#ffffff",
}


@dataclass
class PlotConfig:
    output_dir: str = "images"
    dpi: int = 300
    format: str = "png"
    figsize: tuple[int, int] = field(default_factory=lambda: (10, 6))
    font_family: str = "serif"

    @classmethod
    def from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> "PlotConfig":
        """Load plot config from a project's config.yaml output section."""
        if config_path is None:
            return cls()
        path = Path(config_path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        out = data.get("output", {})
        figsize_raw = out.get("figsize", [10, 6])
        return cls(
            output_dir=str(out.get("figures_dir", "images")),
            dpi=int(out.get("figure_dpi", 300)),
            format=str(out.get("figure_format", "png")),
            figsize=tuple(figsize_raw),
            font_family=str(out.get("font_family", "serif")),
        )

    @classmethod
    def from_script(cls, script_path: Union[str, Path]) -> "PlotConfig":
        """Load config from config.yaml in the same directory as the calling script."""
        config_yaml = Path(script_path).parent / "config.yaml"
        return cls.from_yaml(config_yaml)


def set_tufte_defaults(config: Optional[PlotConfig] = None) -> None:
    """Apply Tufte-style rcParams globally."""
    font_family = config.font_family if config else "serif"
    dpi = config.dpi if config else 300
    mpl.rcParams.update({
        "font.family": font_family,
        "font.serif": ["Palatino", "Times New Roman", "Times", "serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })


def apply_tufte_style(ax: plt.Axes, show_grid: bool = False) -> plt.Axes:
    """Remove top/right spines and optionally disable grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_grid:
        ax.grid(False)
    return ax


def setup_tufte_plot(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    fontsize: int = 11,
) -> plt.Axes:
    """Configure axis labels and apply Tufte styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontweight="normal", fontsize=fontsize + 1)
    return apply_tufte_style(ax)


def save_tufte_figure(
    filename: str,
    config: Optional[PlotConfig] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Save current figure and close it.

    Output directory priority: output_dir arg > config.output_dir > "images"
    DPI comes from config if provided, otherwise matplotlib's savefig.dpi rcParam.
    """
    if config is None:
        config = PlotConfig()
    out = Path(output_dir or config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fp = Path(filename)
    if not fp.suffix:
        filename = f"{filename}.{config.format}"
    filepath = out / Path(filename).name

    plt.tight_layout()
    plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    return filepath


def save_fig(
    filename: str,
    config: Optional[PlotConfig] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Alias for save_tufte_figure — used in analysis scripts."""
    return save_tufte_figure(filename, config=config, output_dir=output_dir)
