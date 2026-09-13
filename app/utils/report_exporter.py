"""
Generates a downloadable PDF report for a single scan. Adapted from Repo 5
(IviweBooi)'s report-export feature.
"""
import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_scan_report_pdf(scan) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#2B2E2A"))
    heading_style = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], textColor=colors.HexColor("#4A6150"))
    body_style = styles["BodyText"]

    story = []
    story.append(Paragraph("AI Content &amp; Website Detector — Scan Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    meta_table = Table(
        [
            ["Scanned on", scan.created_at.strftime("%Y-%m-%d %H:%M UTC")],
            ["Input type", scan.input_type],
            ["Input", scan.input_summary],
            ["Overall score", f"{scan.overall_score} / 100"],
            ["Classification band", scan.band],
        ],
        colWidths=[1.7 * inch, 4.5 * inch],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#565A52")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E9EAE2")),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Engine Breakdown", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    for engine in scan.engine_results():
        story.append(Paragraph(f"<b>{engine['engine_name']}</b> — score {engine['score']} ({engine['confidence']} confidence)", body_style))
        story.append(Paragraph(engine["summary"], body_style))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Spacer(1, 0.2 * inch))
    disclaimer = (
        "Thresholds used by these detection engines are inherited from their source methods and "
        "have not yet been validated against an independently labeled dataset. Treat this report as "
        "a strong signal to investigate further, not a definitive verdict."
    )
    story.append(Paragraph(f"<i>{disclaimer}</i>", styles["Italic"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))

    doc.build(story)
    return buffer.getvalue()
