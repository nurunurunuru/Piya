"""
src/guidance/report_generator.py
Generates PDF and HTML planning reports for Modish City guidance.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dhaka Building Height Analysis — Modish City Report</title>
  <style>
    :root {
      --primary: #1a3a5c; --accent: #e63946; --light: #f1faee;
      --success: #2a9d8f; --warn: #e9c46a; --danger: #e76f51;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa;
           color: #212529; line-height: 1.6; }
    header { background: var(--primary); color: white; padding: 40px 60px; }
    header h1 { font-size: 2rem; margin-bottom: 8px; }
    header p  { opacity: 0.8; font-size: 0.95rem; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
             font-size: 0.78rem; font-weight: bold; color: white; }
    .badge-Critical { background: var(--danger); }
    .badge-High     { background: var(--accent); }
    .badge-Medium   { background: var(--warn); color: #333; }
    .badge-Low      { background: var(--success); }
    main { max-width: 1100px; margin: 40px auto; padding: 0 30px; }
    section { margin-bottom: 50px; }
    h2 { color: var(--primary); border-bottom: 3px solid var(--accent);
         padding-bottom: 8px; margin-bottom: 20px; font-size: 1.4rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: var(--primary); color: white; padding: 10px 12px; text-align: left; }
    td { padding: 9px 12px; border-bottom: 1px solid #dee2e6; }
    tr:nth-child(even) { background: #f1f3f5; }
    .rec-card { background: white; border-left: 5px solid var(--accent);
                border-radius: 6px; padding: 18px 22px; margin-bottom: 18px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .rec-card h3 { font-size: 1rem; margin-bottom: 8px; color: var(--primary); }
    .rec-card p  { color: #495057; font-size: 0.9rem; margin-bottom: 10px; }
    .rec-card ul { padding-left: 20px; color: #495057; font-size: 0.88rem; }
    .rec-card .impact { margin-top: 10px; font-style: italic;
                        color: var(--success); font-size: 0.85rem; }
    footer { text-align: center; padding: 30px; background: var(--primary);
             color: rgba(255,255,255,0.6); font-size: 0.82rem; margin-top: 60px; }
  </style>
</head>
<body>
<header>
  <h1>🏙️ Dhaka Building Height Analysis — Modish City Guidance Report</h1>
  <p>Generated: {{ generated_at }} &nbsp;|&nbsp; Total Buildings Analyzed: {{ total_buildings }} &nbsp;|&nbsp; Zones Covered: {{ n_zones }}</p>
</header>

<main>

  <section>
    <h2>📊 Zone-wise Height Statistics</h2>
    <table>
      <tr>
        <th>Zone</th><th>Buildings</th><th>Mean Height (m)</th>
        <th>Max Height (m)</th><th>Violations</th><th>Violation %</th>
      </tr>
      {% for _, row in zone_summary.iterrows() %}
      <tr>
        <td><strong>{{ row.zone_name }}</strong></td>
        <td>{{ row.total_buildings }}</td>
        <td>{{ "%.1f"|format(row.mean_height) }}</td>
        <td>{{ "%.1f"|format(row.max_height) }}</td>
        <td>{{ row.violations }}</td>
        <td>{{ "%.1f"|format(row.violation_pct) }}%</td>
      </tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>⚠️ Risk Summary</h2>
    <table>
      <tr><th>Zone</th><th>Low</th><th>Moderate</th><th>High</th><th>Critical</th><th>Total</th></tr>
      {% for zone, row in risk_matrix.iterrows() if zone != 'All' %}
      <tr>
        <td><strong>{{ zone }}</strong></td>
        <td>{{ row.get('Low', 0) }}</td>
        <td>{{ row.get('Moderate', 0) }}</td>
        <td>{{ row.get('High', 0) }}</td>
        <td>{{ row.get('Critical', 0) }}</td>
        <td>{{ row.get('All', 0) }}</td>
      </tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>🌆 Modish City Recommendations</h2>
    {% for rec in recommendations %}
    <div class="rec-card">
      <div style="margin-bottom:8px;">
        <span class="badge badge-{{ rec.priority }}">{{ rec.priority }}</span>
        <span style="margin-left:8px; font-size:0.8rem; color:#6c757d;">{{ rec.category }} — {{ rec.zone_name }}</span>
      </div>
      <h3>{{ rec.title }}</h3>
      <p>{{ rec.description }}</p>
      <ul>
        {% for action in rec.action_items %}
        <li>{{ action }}</li>
        {% endfor %}
      </ul>
      {% if rec.estimated_impact %}
      <div class="impact">💡 Impact: {{ rec.estimated_impact }}</div>
      {% endif %}
    </div>
    {% endfor %}
  </section>

</main>
<footer>
  Dhaka Building Height Analysis | Urban AI Lab | Data: Sentinel-2, OSM, BNBC Standards
</footer>
</body>
</html>
"""


class ReportGenerator:
    """Generates HTML and optionally PDF reports."""

    def generate_html(
        self,
        zone_summary: pd.DataFrame,
        risk_matrix: pd.DataFrame,
        recommendations: list,
        output_path: str,
        total_buildings: int = 0,
    ) -> str:
        """Render the HTML report from template."""
        from jinja2 import Environment

        env = Environment()
        template = env.from_string(REPORT_HTML_TEMPLATE)

        html = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_buildings=total_buildings,
            n_zones=len(zone_summary),
            zone_summary=zone_summary,
            risk_matrix=risk_matrix,
            recommendations=recommendations,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report saved → {output_path}")
        return output_path

    def generate_pdf(self, html_path: str, pdf_path: str) -> str:
        """Convert HTML report to PDF using weasyprint."""
        try:
            import weasyprint
            weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
            logger.info(f"PDF report saved → {pdf_path}")
        except ImportError:
            logger.warning("weasyprint not installed. Skipping PDF generation.")
            logger.warning("Install with: pip install weasyprint")
        return pdf_path
