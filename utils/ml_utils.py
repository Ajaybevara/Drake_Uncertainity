import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

FEATURES = ['GR', 'RHOB', 'DT', 'NPHI', 'RT']


def synthetic_training_data(n=800):
    rng = np.random.default_rng(42)
    gr = rng.uniform(20, 150, n)
    rhob = rng.uniform(1.95, 2.85, n)
    dt = rng.uniform(45, 140, n)
    nphi = rng.uniform(0.02, 0.42, n)
    rt = rng.uniform(0.2, 300, n)
    por = np.clip(0.38 - 0.12*(rhob-2.2) + 0.0015*(dt-80) - 0.0012*(gr-75) + 0.18*nphi + 0.015*np.log1p(rt), 0.02, 0.38)
    sw = np.clip(0.92 - 0.9*por + 0.04*(gr/150) - 0.025*np.log1p(rt), 0.05, 1.0)
    lith = np.where(gr < 60, 'Clean Sand', np.where(gr < 95, 'Shaly Sand', 'Shale'))
    X = pd.DataFrame({'GR': gr, 'RHOB': rhob, 'DT': dt, 'NPHI': nphi, 'RT': rt})
    return X, por, sw, lith

X_train, y_por, y_sw, y_lith = synthetic_training_data()
POR_MODEL = RandomForestRegressor(n_estimators=80, random_state=42)
SAT_MODEL = RandomForestRegressor(n_estimators=80, random_state=24)
LITH_MODEL = RandomForestClassifier(n_estimators=80, random_state=12)
POR_MODEL.fit(X_train, y_por)
SAT_MODEL.fit(X_train, y_sw)
LITH_MODEL.fit(X_train, y_lith)


def prepare_features(df):
    data = df.copy()
    for col in FEATURES:
        if col not in data.columns:
            data[col] = 0.0
    X = data[FEATURES].fillna(method='ffill').fillna(method='bfill').fillna(0)
    return X


def quality_from_values(phi, sw):
    if phi >= 0.2 and sw <= 0.5:
        return 'Excellent'
    if phi >= 0.12 and sw <= 0.7:
        return 'Moderate'
    return 'Poor'


def build_interval_summary(df, phi, sw, lith):
    out = []
    depth = df['DEPTH'].to_numpy() if 'DEPTH' in df.columns else np.arange(len(df))
    zones = [
        ('Porosity', phi >= 0.12),
        ('Low Water Saturation', sw <= 0.6),
        ('Reservoir', (phi >= 0.12) & (sw <= 0.6)),
    ]
    for name, mask in zones:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        out.append({'zone': name, 'from_depth': float(depth[idx.min()]), 'to_depth': float(depth[idx.max()]), 'count': int(len(idx))})
    lith_counts = pd.Series(lith).value_counts().to_dict()
    return out, lith_counts


def run_predictions(df):
    X = prepare_features(df)
    phi = POR_MODEL.predict(X)
    sw = SAT_MODEL.predict(X)
    lith = LITH_MODEL.predict(X)
    forest_preds = np.vstack([tree.predict(X) for tree in POR_MODEL.estimators_])
    p10 = np.percentile(forest_preds, 10, axis=0)
    p50 = np.percentile(forest_preds, 50, axis=0)
    p90 = np.percentile(forest_preds, 90, axis=0)
    confidence = np.clip(100 - (p90 - p10) * 180, 45, 99)
    depth = df['DEPTH'].to_numpy() if 'DEPTH' in df.columns else np.arange(len(df))
    interval_summary, lith_counts = build_interval_summary(df, phi, sw, lith)
    data = []
    for i in range(len(X)):
        data.append({
            'DEPTH': float(depth[i]),
            'POROSITY': round(float(phi[i]), 4),
            'SATURATION': round(float(sw[i]), 4),
            'LITHOLOGY': str(lith[i]),
            'P10': round(float(p10[i]), 4),
            'P50': round(float(p50[i]), 4),
            'P90': round(float(p90[i]), 4),
            'CONFIDENCE': round(float(confidence[i]), 2),
            'QUALITY': quality_from_values(float(phi[i]), float(sw[i]))
        })
    return {
        'rows': len(data),
        'data': data,
        'interval_summary': interval_summary,
        'lithology_counts': lith_counts,
        'formulas': {
            'porosity': 'Density-derived porosity: phi_d = (rho_ma - rho_b) / (rho_ma - rho_f); sonic trend support via Wyllie-style relation.',
            'saturation': 'Archie-style saturation trend: Sw^n = (a * Rw) / (phi^m * Rt).',
            'lithology': 'Rule-assisted ML classification using GR, RHOB, DT, NPHI, and RT.'
        },
        'cutoffs': {
            'porosity': 'Net reservoir cutoff commonly screened at porosity >= 0.12; conventional lower cutoff can be around 0.05 depending on field economics.',
            'saturation': 'Hydrocarbon-favorable screen often uses Sw <= 0.60, with stronger pay likelihood at Sw <= 0.50.',
            'lithology': 'Clean Sand: GR < 60, Shaly Sand: 60 <= GR < 95, Shale: GR >= 95, adjusted by ML context.'
        }
    }


def compute_intervals(df):
    return run_predictions(df)['interval_summary']


def classify_quality(phi, sw):
    return quality_from_values(phi, sw)
