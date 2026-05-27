# DrakeAI Dashboard Plus
This version matches the requested industrial dashboard style more closely.

## Main additions
- Dashboard hero text: Industrial Geological Dashboard
- Metrics cards: Wells processed, Predictions, Reservoir score, Uncertainty index
- Activity stream and Recent uploads panels
- River-style uncertainty curves using Plotly fill bands

## Run
```bash
pip install -r requirements.txt
python app.py
```


Dynamic LAS enhancement update: dashboard upload now parses uploaded LAS files directly, extracts all headers and curves dynamically, renders available logs and statistics, keeps analysis history, and supports CSV/Excel/PDF/JSON exports.
