import io
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def make_json_bytes(payload):
    return json.dumps(payload, indent=2).encode('utf-8')


def make_excel_bytes(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='DrakeAI Results', index=False)
    bio.seek(0)
    return bio.read()


def make_pdf_bytes(well_info, pred):
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    y = 800
    c.setFont('Helvetica-Bold', 16)
    c.drawString(50, y, 'DrakeAI Reservoir Intelligence Report')
    y -= 30
    c.setFont('Helvetica', 10)
    for k, v in well_info.items():
        c.drawString(50, y, f'{k}: {v}')
        y -= 15
        if y < 80:
            c.showPage(); y = 800
    y -= 10
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y, 'Prediction Summary')
    y -= 20
    c.setFont('Helvetica', 10)
    for zone in pred.get('interval_summary', []):
        c.drawString(50, y, f"{zone['zone']}: {zone['from_depth']:.2f} to {zone['to_depth']:.2f} ({zone['count']} samples)")
        y -= 15
        if y < 80:
            c.showPage(); y = 800
    c.save()
    bio.seek(0)
    return bio.read()
