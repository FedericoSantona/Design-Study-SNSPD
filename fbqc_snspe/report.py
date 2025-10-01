"""Lightweight report generation for SNSPD design studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from jinja2 import Environment, BaseLoader

from .objectives import BandMetrics

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SNSPD Design Study</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
        th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }
        th { background-color: #f5f5f5; }
        h1, h2, h3 { font-family: 'Helvetica Neue', Arial, sans-serif; }
    </style>
</head>
<body>
    <h1>SNSPD Design Study</h1>
    <h2>Summary</h2>
    <p>Delta A<sub>max</sub>: {{ metrics.delta_db_max | round(3) }} dB</p>
    <p>Mean absorptance: {{ metrics.mean_absorptance | round(3) }}</p>
    <p>Worst-case absorptance: {{ metrics.worst_case_absorptance | round(3) }}</p>
    <p>Band condition satisfied: {{ metrics.band_ok }}</p>

    <h2>Top Candidates</h2>
    {{ table_html | safe }}

    {% if figures %}
    <h2>Figures</h2>
    {% for figure_path in figures %}
        <div>
            <h3>{{ figure_path.name }}</h3>
            <img src="{{ figure_path }}" style="max-width: 100%;" />
        </div>
    {% endfor %}
    {% endif %}
</body>
</html>
"""


def render_report(output: Path, metrics: BandMetrics, candidates: pd.DataFrame, figures: Iterable[Path] | None = None) -> Path:
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(TEMPLATE)
    table_html = candidates.to_html(index=False, float_format="{:.3f}".format)
    html = template.render(metrics=metrics, table_html=table_html, figures=list(figures or []))
    output.write_text(html)
    return output


__all__ = ["render_report"]
