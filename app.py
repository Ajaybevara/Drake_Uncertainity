import io
import zipfile
import json
import math
import uuid
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
REPORT_DIR = BASE_DIR / 'reports'
DATABASE_DIR = BASE_DIR / 'database'
PRED_CACHE_DIR = BASE_DIR / 'database' / 'pred_cache'
for p in [UPLOAD_DIR, REPORT_DIR, DATABASE_DIR, PRED_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'drakeai-secret-key'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

USERS_FILE = DATABASE_DIR / 'users.json'
ANALYSIS_FILE = DATABASE_DIR / 'analysis_history.json'
TASKS_FILE = DATABASE_DIR / 'processing_tasks.json'

WELL_INFO_FIELDS = [
    ('WELL', 'Well Name'), ('COMP', 'Company'), ('FLD', 'Field'), ('LOC', 'Location'),
    ('CNTY', 'County'), ('STAT', 'State'), ('CTRY', 'Country'), ('OPER', 'Operator'),
    ('SRVC', 'Service Company'), ('API', 'API Number'), ('UWI', 'UWI'), ('WID', 'Well ID'),
    ('LATI', 'Latitude'), ('LONG', 'Longitude'), ('STRT', 'Start Depth'), ('STOP', 'Stop Depth'),
    ('STEP', 'Step Size'), ('NULL', 'Null Value'), ('VERS', 'Version'), ('DATE', 'Date')
]


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return default
    return default


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_users():
    return load_json(USERS_FILE, {'users': []})


def save_users(data):
    save_json(USERS_FILE, data)


def load_history_store():
    return load_json(ANALYSIS_FILE, {'items': []})


def save_history_store(data):
    save_json(ANALYSIS_FILE, data)

def load_tasks_store():
    return load_json(TASKS_FILE, {'tasks': {}})

def save_tasks_store(data):
    save_json(TASKS_FILE, data)

def set_task(task_id, payload):
    store = load_tasks_store()
    store['tasks'][task_id] = payload
    save_tasks_store(store)

def get_task(task_id):
    store = load_tasks_store()
    return store.get('tasks', {}).get(task_id)

def update_task(task_id, **updates):
    store = load_tasks_store()
    task = store.get('tasks', {}).get(task_id, {})
    task.update(updates)
    store['tasks'][task_id] = task
    save_tasks_store(store)


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user_email'):
            session['user_email'] = 'guest@drakeai.local'
        return fn(*args, **kwargs)
    return wrapper


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'las'


def safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value






def find_log_name(log_names, candidates):
    """Find the actual column name from candidates using 4-pass matching:
    1) Exact  2) Log starts with candidate (NPHISS→NPHI)
    3) Candidate starts with log  4) Candidate inside log name
    Skips DEPTH column always."""
    upper_map = {str(name).upper(): name for name in (log_names or []) if str(name).upper() != 'DEPTH'}
    # Pass 1: exact
    for cand in candidates:
        cu = str(cand).upper()
        if cu in upper_map:
            return upper_map[cu]
    # Pass 2: log name starts with candidate (e.g. NPHISS starts with NPHI)
    for cand in candidates:
        cu = str(cand).upper()
        for key, orig in upper_map.items():
            if key.startswith(cu):
                return orig
    # Pass 3: candidate starts with log name (e.g. candidate=NPHISS, log=NPL)
    for cand in candidates:
        cu = str(cand).upper()
        for key, orig in upper_map.items():
            if cu.startswith(key) and len(key) >= 2:
                return orig
    # Pass 4: candidate appears anywhere in log name
    for cand in candidates:
        cu = str(cand).upper()
        for key, orig in upper_map.items():
            if cu in key and len(cu) >= 2:
                return orig
    return None


def clean_numeric_series(df, col):
    if not col or col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype='float64')
    series = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float64')
    return series


def clip01(series):
    return pd.to_numeric(series, errors='coerce').clip(0, 1)


def safe_sqrt(value):
    arr = np.asarray(value, dtype=float)
    arr[arr < 0] = np.nan
    return np.sqrt(arr)


def predict_vsh_ai(df):
    """Calculate VSH using Linear IGR method from GR log."""
    gr_col = find_log_name(list(df.columns), ['GRD','GR','GRS','GRR','CGR','SGR','HSGR','GRC','GAMMA','GAMMARAY'])
    if not gr_col or gr_col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype='float64')
    gr = pd.to_numeric(df[gr_col], errors='coerce').astype('float64')
    gr_valid = gr.dropna()
    if gr_valid.empty:
        return pd.Series([np.nan] * len(df), index=df.index, dtype='float64')
    gr_min = float(gr_valid.quantile(0.05))
    gr_max = float(gr_valid.quantile(0.95))
    denom = (gr_max - gr_min) if gr_max != gr_min else np.nan
    igr = ((gr - gr_min) / denom).clip(0.0, 1.0)
    return pd.to_numeric(igr, errors='coerce').astype('float64')


def predict_porosity_ai(df):
    """Calculate total porosity from density log (Density Porosity method)."""
    rhob_col = find_log_name(list(df.columns), ['RHOB','RHOZ','DEN','ZDEN'])
    if not rhob_col or rhob_col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype='float64')
    rhob = pd.to_numeric(df[rhob_col], errors='coerce').astype('float64')
    rhoma, rhof = 2.65, 1.0
    phit = ((rhoma - rhob) / (rhoma - rhof)).clip(0.0, 1.0)
    return pd.to_numeric(phit, errors='coerce').astype('float64')


def predict_saturation_ai(df):
    """Calculate water saturation using Archie's equation."""
    rt_col = find_log_name(list(df.columns), ['RT','RESD','ILD','LLD','AT90','RDEP'])
    rhob_col = find_log_name(list(df.columns), ['RHOB','RHOZ','DEN'])
    if not rt_col or rt_col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype='float64')
    rt = pd.to_numeric(df[rt_col], errors='coerce').astype('float64').where(lambda x: x > 0)
    phit = predict_porosity_ai(df)
    phie = phit.clip(0.001, 1.0)
    rw, a, m, n = 0.1, 1.0, 2.0, 2.0
    sw = (((a * rw) / ((phie ** m) * rt)) ** (1.0 / n)).clip(0.0, 1.0)
    sw = sw.replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(sw, errors='coerce').astype('float64')


def predict_lithology_ai(df):
    """Classify lithology using VSH + RHOB rule-based logic."""
    vsh = predict_vsh_ai(df)
    rhob_col = find_log_name(list(df.columns), ['RHOB','RHOZ','DEN'])
    rhob = clean_numeric_series(df, rhob_col) if rhob_col else pd.Series([np.nan]*len(df), index=df.index)
    phit = predict_porosity_ai(df)
    labels = []
    for i in range(len(df)):
        v_i = vsh.iloc[i]; r_i = rhob.iloc[i]; p_i = phit.iloc[i]
        if pd.notna(r_i) and r_i < 1.80:
            labels.append('Coal')
        elif pd.notna(r_i) and 2.80 <= r_i <= 2.90:
            labels.append('Dolomite')
        elif pd.notna(r_i) and 2.68 <= r_i <= 2.75:
            labels.append('Limestone')
        elif pd.notna(v_i) and v_i > 0.50:
            labels.append('Shale')
        elif pd.notna(v_i) and v_i > 0.30:
            labels.append('Shaly Sand')
        elif pd.notna(v_i) and v_i <= 0.30 and pd.notna(p_i) and p_i >= 0.10:
            labels.append('Clean Sandstone')
        else:
            labels.append('Unknown')
    return pd.Series(labels, index=df.index, dtype='object')


def compute_prediction_sections(item, config=None):
    config = config or {}
    df = pd.DataFrame(item.get('logs_data', []))
    if df.empty:
        return {'success': False, 'message': 'No log data available.'}
    if 'DEPTH' not in df.columns:
        df['DEPTH'] = np.arange(len(df), dtype=float)
    df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce').ffill().bfill().fillna(0)
    log_names = item.get('log_names', [])

    defaults = {
        'vsh': {
            'gr_log': find_log_name(log_names, ['GRD', 'GR', 'GRC', 'GAMMA']),
            'gr_min': 20.0,
            'gr_max': 120.0,
            'method': 'linear'
        },
        'porosity': {
            'method': 'density',
            'rhob_log': find_log_name(log_names, ['RHOB', 'RHOZ', 'DEN']),
            'nphi_log': find_log_name(log_names, ['NPHI', 'NPHIS', 'NPHISS', 'NPL']),
            'dt_log': find_log_name(log_names, ['DT', 'DTP', 'AC', 'SONIC']),
            'rhoma': 2.65,
            'rhof': 1.0,
            'dtma': 55.5,
            'dtfl': 189.0,
            'nphi_unit': 'fraction'
        },
        'saturation': {
            'method': 'archie',
            'rt_log': find_log_name(log_names, ['RT', 'RESD', 'ILD', 'LLD', 'AT90', 'HDRS', 'RDEP']),
            'rw': 0.10,
            'rsh': 2.0,
            'a': 1.0,
            'm': 2.0,
            'n': 2.0
        },
        'lithology': {
            'gr_or_vsh_log': 'VSH',
            'rhob_log': find_log_name(log_names, ['RHOB', 'RHOZ', 'DEN']),
            'nphi_log': find_log_name(log_names, ['NPHI', 'NPHIS', 'NPHISS', 'NPL']),
            'dt_log': find_log_name(log_names, ['DT', 'DTP', 'AC', 'SONIC']),
            'pe_log': find_log_name(log_names, ['PE', 'PEF']),
            'clean_vsh_max': 0.30,
            'shaly_vsh_max': 0.50,
            'phie_min': 0.10,
            'sand_rhob_min': 2.55,
            'sand_rhob_max': 2.70,
            'lime_rhob_min': 2.68,
            'lime_rhob_max': 2.75,
            'dolo_rhob_min': 2.80,
            'dolo_rhob_max': 2.90,
            'coal_rhob_max': 1.80,
            'lime_pe': 5.0,
            'dolo_pe': 3.0,
            'pe_tolerance': 0.7
        }
    }

    for section, vals in defaults.items():
        defaults[section].update(config.get(section, {}))

    # Coerce ALL numeric config values to float (HTML inputs always send strings)
    _FLOAT_FIELDS = {
        'vsh':        ['gr_min','gr_max'],
        'porosity':   ['rhoma','rhof','dtma','dtfl'],
        'saturation': ['rw','rsh','a','m','n'],
        'lithology':  ['clean_vsh_max','shaly_vsh_max','phie_min',
                       'sand_rhob_min','sand_rhob_max','lime_rhob_min','lime_rhob_max',
                       'dolo_rhob_min','dolo_rhob_max','coal_rhob_max',
                       'lime_pe','dolo_pe','pe_tolerance'],
    }
    for sec, fields in _FLOAT_FIELDS.items():
        for fld in fields:
            defaults[sec][fld] = safe_float(defaults[sec].get(fld)) if defaults[sec].get(fld) is not None else defaults[sec].get(fld)

    out = df[['DEPTH']].copy()

    vcfg = defaults['vsh']
    # Resolve gr_log: try config value first (case-insensitive), then find_log_name fallback
    col_map = {c.upper(): c for c in df.columns}
    gr_log_name = vcfg.get('gr_log') or ''
    # Always do case-insensitive match first
    if gr_log_name:
        gr_log_name = col_map.get(gr_log_name.upper(), '')
    # If still not found, try fuzzy matching against all common GR aliases
    if not gr_log_name:
        gr_log_name = find_log_name(list(df.columns), [
            'GRD','GR','GRS','GRR','CGR','SGR','HSGR','GRC','GAMMA','GAMMARAY','GR_RAW'
        ]) or ''
    # Final case-insensitive column match
    gr_log_name = col_map.get(gr_log_name.upper(), gr_log_name)
    gr = clean_numeric_series(df, gr_log_name).astype('float64')
    gr_min = safe_float(vcfg.get('gr_min'))
    gr_max = safe_float(vcfg.get('gr_max'))
    if gr_min is None and gr.notna().any(): gr_min = float(gr.min())
    if gr_max is None and gr.notna().any(): gr_max = float(gr.max())
    igr_raw = ((gr - gr_min) / ((gr_max - gr_min) if (gr_max is not None and gr_min is not None and gr_max != gr_min) else np.nan)).replace([np.inf, -np.inf], np.nan)
    igr = pd.to_numeric(igr_raw, errors='coerce').astype('float64').clip(0.0, 1.0)
    method = str(vcfg.get('method', 'linear')).lower()
    if method == 'larionov_tertiary':
        vsh = pd.to_numeric(0.083 * ((2.0 ** (3.7 * igr)) - 1.0), errors='coerce').astype('float64')
    elif method == 'larionov_older':
        vsh = pd.to_numeric(0.33 * ((2.0 ** (2.0 * igr)) - 1.0), errors='coerce').astype('float64')
    elif method == 'clavier':
        vsh = pd.to_numeric(1.7 - safe_sqrt(3.38 - ((igr + 0.7) ** 2)), errors='coerce').astype('float64')
    elif method == 'steiber':
        vsh = pd.to_numeric(igr / (3.0 - 2.0 * igr), errors='coerce').astype('float64')
    else:
        method = 'linear'
        vsh = igr.copy()
    out['GR'] = gr
    out['IGR'] = igr.clip(0.0, 1.0)
    out['VSH'] = pd.to_numeric(vsh, errors='coerce').astype('float64').clip(0.0, 1.0)

    pcfg = defaults['porosity']
    phit = pd.Series([np.nan] * len(df), index=df.index, dtype='float64')
    pmethod = str(pcfg.get('method', 'density')).lower()
    # Build case-insensitive column resolver for all log references
    _col_map = {c.upper(): c for c in df.columns}
    def _resolve_col(cfg_key, candidates):
        name = pcfg.get(cfg_key) or ''
        if name and name.upper() in _col_map:
            return _col_map[name.upper()]
        return find_log_name(list(df.columns), candidates) or ''
    if pmethod == 'density':
        rhob = clean_numeric_series(df, _resolve_col('rhob_log', ['RHOB','RHOZ','DEN'])).astype('float64')
        rhoma = safe_float(pcfg.get('rhoma')) or 2.65
        rhof = safe_float(pcfg.get('rhof')) or 1.0
        phit = pd.to_numeric((rhoma - rhob) / ((rhoma - rhof) if rhoma != rhof else np.nan), errors='coerce').astype('float64')
    elif pmethod == 'neutron':
        nphi = clean_numeric_series(df, _resolve_col('nphi_log', ['NPHI','NPHIS','NPHISS','NPL'])).astype('float64')
        phit = (nphi / 100.0 if str(pcfg.get('nphi_unit', 'fraction')).lower() == 'percent' else nphi).astype('float64')
    elif pmethod == 'sonic':
        dt = clean_numeric_series(df, _resolve_col('dt_log', ['DT','DTP','AC','SONIC'])).astype('float64')
        dtma = safe_float(pcfg.get('dtma')) or 55.5
        dtfl = safe_float(pcfg.get('dtfl')) or 189.0
        phit = pd.to_numeric((dt - dtma) / ((dtfl - dtma) if dtfl != dtma else np.nan), errors='coerce').astype('float64')
    elif pmethod == 'density_neutron':
        rhob = clean_numeric_series(df, _resolve_col('rhob_log', ['RHOB','RHOZ','DEN'])).astype('float64')
        nphi = clean_numeric_series(df, _resolve_col('nphi_log', ['NPHI','NPHIS','NPHISS','NPL'])).astype('float64')
        rhoma = safe_float(pcfg.get('rhoma')) or 2.65
        rhof = safe_float(pcfg.get('rhof')) or 1.0
        phid = pd.to_numeric((rhoma - rhob) / ((rhoma - rhof) if rhoma != rhof else np.nan), errors='coerce').astype('float64')
        phin = (nphi / 100.0 if str(pcfg.get('nphi_unit', 'fraction')).lower() == 'percent' else nphi).astype('float64')
        # Correct industry-standard RMS (root-mean-square) average for density-neutron crossplot
        phit = pd.to_numeric(np.sqrt((phid.fillna(0.0)**2 + phin.fillna(0.0)**2) / 2.0), errors='coerce').astype('float64')
        # Propagate NaN where both inputs are NaN
        both_nan = phid.isna() & phin.isna()
        phit = phit.where(~both_nan, other=np.nan)
    out['PHIT'] = pd.to_numeric(phit, errors='coerce').astype('float64').clip(0.0, 1.0)
    out['PHIE'] = pd.to_numeric(out['PHIT'] * (1.0 - out['VSH'].astype('float64')), errors='coerce').astype('float64').clip(0.0, 1.0)

    scfg = defaults['saturation']
    _scol_map = {c.upper(): c for c in df.columns}
    _rt_name = scfg.get('rt_log') or ''
    if _rt_name and _rt_name.upper() in _scol_map:
        _rt_name = _scol_map[_rt_name.upper()]
    else:
        _rt_name = find_log_name(list(df.columns), ['RT','RESD','ILD','LLD','AT90','HDRS','RDEP']) or ''
    rt = clean_numeric_series(df, _rt_name)
    rt = rt.where(rt > 0)
    rw = safe_float(scfg.get('rw')) or 0.1
    rsh = safe_float(scfg.get('rsh')) or 2.0
    a = safe_float(scfg.get('a')) or 1.0
    m = safe_float(scfg.get('m')) or 2.0
    n = safe_float(scfg.get('n')) or 2.0
    # Ensure VSH and PHIE are strictly float64 to avoid numpy ufunc type errors
    vsh_f = pd.to_numeric(out['VSH'], errors='coerce').astype('float64')
    phie_f = pd.to_numeric(out['PHIE'], errors='coerce').astype('float64')
    rt_f = pd.to_numeric(rt, errors='coerce').astype('float64')
    phie_safe = phie_f.where(phie_f > 0)
    sw_archie = (((a * rw) / ((phie_safe ** m) * rt_f)) ** (1.0 / n)).replace([np.inf, -np.inf], np.nan)
    term_a = (vsh_f.fillna(0.0) ** (1.0 - vsh_f.fillna(0.0) / 2.0)) / float(np.sqrt(max(rsh, 1e-6)))
    term_b = pd.Series(np.sqrt((phie_safe.fillna(0.0).values ** m) / max(a * rw, 1e-6)), index=df.index, dtype='float64')
    sw_ind_raw = ((1.0 / np.sqrt(rt_f)) - term_a) / term_b.replace(0.0, np.nan)
    sw_ind_raw = pd.to_numeric(sw_ind_raw, errors='coerce').astype('float64')
    sw_ind = (sw_ind_raw.clip(lower=0.0) ** (2.0 / n))
    sw_ind = sw_ind.replace([np.inf, -np.inf], np.nan)
    smethod = str(scfg.get('method', 'archie')).lower()
    sat_method = []
    sw_values = []
    for i in range(len(df)):
        v = vsh_f.iloc[i]
        sa = sw_archie.iloc[i]
        si = sw_ind.iloc[i]
        active = smethod
        val = sa
        if smethod == 'indonesia':
            active = 'indonesia'; val = si
        elif smethod == 'auto':
            if pd.notna(v) and v <= 0.15:
                active = 'archie'; val = sa
            else:
                active = 'indonesia'; val = si
        else:
            active = 'archie'; val = sa
        sat_method.append(active.title())
        sw_values.append(val)
    out['SW'] = clip01(pd.Series(sw_values, index=df.index, dtype='float64'))
    out['SATURATION_METHOD'] = sat_method

    lcfg = defaults['lithology']
    _lcol_map = {c.upper(): c for c in df.columns}
    def _lresolve(key, cands):
        name = lcfg.get(key) or ''
        if name and name.upper() in _lcol_map:
            return _lcol_map[name.upper()]
        return find_log_name(list(df.columns), cands) or ''
    rhob_l = clean_numeric_series(df, _lresolve('rhob_log', ['RHOB','RHOZ','DEN']))
    nphi_l = clean_numeric_series(df, _lresolve('nphi_log', ['NPHI','NPHIS','NPHISS','NPL']))
    pe_l   = clean_numeric_series(df, _lresolve('pe_log',   ['PE','PEF']))
    lith = []
    for i in range(len(df)):
        vsh_i = out['VSH'].iloc[i]
        phie_i = out['PHIE'].iloc[i]
        rhob_i = rhob_l.iloc[i]
        nphi_i = nphi_l.iloc[i]
        pe_i = pe_l.iloc[i]
        label = 'Unknown'
        if pd.notna(rhob_i) and rhob_i < lcfg['coal_rhob_max']:
            label = 'Coal'
        elif pd.notna(rhob_i) and lcfg['dolo_rhob_min'] <= rhob_i <= lcfg['dolo_rhob_max'] and (pd.isna(pe_i) or abs(pe_i - lcfg['dolo_pe']) <= lcfg['pe_tolerance']):
            label = 'Dolomite'
        elif pd.notna(rhob_i) and lcfg['lime_rhob_min'] <= rhob_i <= lcfg['lime_rhob_max'] and (pd.isna(pe_i) or abs(pe_i - lcfg['lime_pe']) <= lcfg['pe_tolerance']) and (pd.isna(nphi_i) or nphi_i <= 0.35):
            label = 'Limestone'
        elif pd.notna(vsh_i) and vsh_i > lcfg['shaly_vsh_max']:
            label = 'Shale'
        elif pd.notna(vsh_i) and vsh_i > lcfg['clean_vsh_max'] and vsh_i <= lcfg['shaly_vsh_max']:
            label = 'Shaly Sand'
        elif pd.notna(vsh_i) and vsh_i <= lcfg['clean_vsh_max'] and pd.notna(phie_i) and phie_i >= lcfg['phie_min'] and (pd.isna(rhob_i) or (lcfg['sand_rhob_min'] <= rhob_i <= lcfg['sand_rhob_max'])):
            label = 'Clean Sandstone'
        lith.append(label)
    out['LITHOLOGY'] = lith
    # Store RT and RHOB in out for export and display
    out['RT'] = pd.to_numeric(rt, errors='coerce').astype('float64')
    out['RHOB'] = rhob_l.values

    return {
        'success': True,
        'config': to_builtin(defaults),
        'vsh_table': to_builtin(out[['DEPTH', 'GR', 'IGR', 'VSH']].replace({np.nan: None}).to_dict(orient='records')),
        'porosity_table': to_builtin(out[['DEPTH', 'PHIT', 'PHIE']].replace({np.nan: None}).to_dict(orient='records')),
        'saturation_table': to_builtin(out[['DEPTH', 'RT', 'SW', 'SATURATION_METHOD']].replace({np.nan: None}).to_dict(orient='records')),
        'lithology_table': to_builtin(out[['DEPTH', 'VSH', 'RHOB', 'LITHOLOGY']].replace({np.nan: None}).to_dict(orient='records')),
        'exports': {
            'vsh': to_builtin(out[['DEPTH', 'GR', 'IGR', 'VSH']].replace({np.nan: None}).to_dict(orient='records')),
            'porosity': to_builtin(out[['DEPTH', 'PHIT', 'PHIE']].replace({np.nan: None}).to_dict(orient='records')),
            'saturation': to_builtin(out[['DEPTH', 'RT', 'SW', 'SATURATION_METHOD']].replace({np.nan: None}).to_dict(orient='records')),
            'lithology': to_builtin(out[['DEPTH', 'VSH', 'RHOB', 'LITHOLOGY']].replace({np.nan: None}).to_dict(orient='records'))
        },
        'warnings': {
            'phie': None if out['VSH'].notna().any() else 'Please calculate Vsh first to compute effective porosity.'
        },
        'active_formulas': {
            'vsh': method,
            'porosity': pmethod,
            'saturation': smethod
        }
    }

def build_prediction_bundle(item):
    df = pd.DataFrame(item.get('logs_data', []))
    if df.empty:
        return {'porosity': [], 'saturation': [], 'lithology': [], 'preview': []}
    if 'DEPTH' not in df.columns:
        df['DEPTH'] = np.arange(len(df), dtype=float)
    depth = pd.to_numeric(df['DEPTH'], errors='coerce').ffill().bfill().fillna(0)
    log_names = item.get('log_names', [])
    available = [c for c in log_names if c in df.columns]
    primary = available[0] if available else None
    secondary = available[1] if len(available) > 1 else primary
    tertiary = available[2] if len(available) > 2 else primary
    base = pd.to_numeric(df[primary], errors='coerce').ffill().bfill() if primary else pd.Series(np.linspace(0, 1, len(df)))
    support = pd.to_numeric(df[secondary], errors='coerce').ffill().bfill() if secondary else base.copy()
    aux = pd.to_numeric(df[tertiary], errors='coerce').ffill().bfill() if tertiary else base.copy()
    base_norm = (base - base.min()) / (base.max() - base.min() + 1e-9)
    support_norm = (support - support.min()) / (support.max() - support.min() + 1e-9)
    aux_norm = (aux - aux.min()) / (aux.max() - aux.min() + 1e-9)
    porosity = (0.08 + 0.22 * (1 - base_norm) + 0.03 * np.sin(depth / 40.0) + 0.02 * (1 - support_norm)).clip(0.02, 0.38)
    sw = (0.18 + 0.68 * base_norm + 0.05 * np.cos(depth / 55.0) + 0.04 * aux_norm).clip(0.05, 0.98)
    spread = (0.03 + 0.06 * base_norm + 0.02 * np.abs(np.sin(depth / 60.0))).clip(0.03, 0.18)
    p10 = (sw - spread).clip(0.01, 0.98)
    p90 = (sw + spread).clip(0.02, 0.99)
    confidence = (97 - (base_norm * 22 + spread * 120)).clip(55, 98)
    reliability = (100 - spread * 180).clip(52, 98)
    lith = np.where(base_norm < 0.33, 'Clean Sand', np.where(base_norm < 0.66, 'Shaly Sand', 'Shale'))
    risk = np.where(reliability >= 85, 'Low', np.where(reliability >= 70, 'Medium', 'High'))
    porosity_rows, saturation_rows, lithology_rows, preview_rows = [], [], [], []
    for i in range(len(df)):
        d = round(float(depth.iloc[i]), 2)
        por = round(float(porosity.iloc[i]), 4)
        sat = round(float(sw.iloc[i]) * 100, 2)
        p10v = round(float(p10.iloc[i]) * 100, 2)
        p90v = round(float(p90.iloc[i]) * 100, 2)
        conf = round(float(confidence.iloc[i]), 2)
        rel = round(float(reliability.iloc[i]), 2)
        lithology = str(lith[i])
        porosity_rows.append({'DEPTH': d, 'POROSITY': por, 'CONFIDENCE': conf})
        saturation_rows.append({'DEPTH': d, 'SW': sat, 'P10': p10v, 'P50': sat, 'P90': p90v, 'RELIABILITY': rel, 'RISK': str(risk[i])})
        lithology_rows.append({'DEPTH': d, 'LITHOLOGY': lithology, 'CONFIDENCE': conf})
        preview_rows.append({'DEPTH': d, 'POROSITY': por, 'WATER_SATURATION': sat, 'LITHOLOGY': lithology, 'CONFIDENCE': conf, 'P10': p10v, 'P50': sat, 'P90': p90v, 'RELIABILITY': rel, 'RISK': str(risk[i])})
    return {'porosity': to_builtin(porosity_rows), 'saturation': to_builtin(saturation_rows), 'lithology': to_builtin(lithology_rows), 'preview': to_builtin(preview_rows)}


def load_current_analysis():
    analysis_id = session.get('analysis_id')
    if not analysis_id:
        return None
    store = load_history_store()
    for item in store.get('items', []):
        if item.get('id') == analysis_id:
            return item
    return None


def save_analysis_item(item):
    store = load_history_store()
    items = [x for x in store.get('items', []) if x.get('id') != item.get('id')]
    items.insert(0, item)
    store['items'] = items[:50]
    save_history_store(store)


def normalize_las_file(file_path: Path) -> Path:
    """
    Some LAS files (e.g. dado1.las, dado2.las) store their data rows under
    ~OTHER INFORMATION instead of the standard ~A / ~ASCII section.
    lasio only recognises ~A, so it reads zero data rows from those files.

    Handles two sub-variants:
      1. ~OTHER header -> column-name row -> numeric data rows
      2. ~OTHER header -> numeric data rows immediately (no column-name row)

    Column-name row detection is broad: allows letters, digits, underscores and
    whitespace, so names like LL8, CILD, SP, DRHO are all matched correctly.
    The key discriminator is that a data row starts with optional whitespace
    followed by an optional minus sign and then a digit.
    """
    import re

    raw = file_path.read_text(encoding='utf-8', errors='replace')

    # ~OTHER … line (mandatory)
    # followed by an OPTIONAL column-name header line:
    #   - starts with a letter (cannot be a data row)
    #   - may contain letters, digits, underscores and whitespace (covers LL8, CILD, etc.)
    # followed by the first numeric data row: spaces, optional -, digit
    other_pattern = re.compile(
        r'(~OTHER[^\n]*\n)'                     # ~OTHER header line  (group 1)
        r'(?:[A-Za-z][A-Za-z0-9 _\t]*\n)?'      # optional column-name row (starts with letter)
        r'(?=[ \t]*-?\d)',                       # lookahead: next content is a data row
        re.IGNORECASE,
    )

    match = other_pattern.search(raw)
    if not match:
        return file_path            # already a standard LAS file – nothing to do

    # Replace only the ~OTHER header line with ~A; keep column names + data intact
    header_end = match.start(1) + len(match.group(1))
    fixed = raw[:match.start(1)] + '~A\n' + raw[header_end:]

    tmp = file_path.with_suffix('.las_tmp')
    tmp.write_text(fixed, encoding='utf-8')
    return tmp

def build_quick_las_preview(file_path: Path, original_filename: str, user_email: str):
    preview = {
        'id': str(uuid.uuid4()),
        'file_name': original_filename,
        'upload_time': datetime.utcnow().isoformat(),
        'analysis_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'user_email': user_email,
        'well_name': file_path.stem,
        'total_logs': 0,
        'file_path': str(file_path),
        'well_info': {'Well Name': file_path.stem, 'Status': 'Quick preview loaded'},
        'available_logs': [],
        'stats': {},
        'summary': {'well_name': file_path.stem, 'total_curves': 0, 'total_samples': 0, 'valid_records': 0, 'missing_values': 0, 'null_values': 0, 'file_size': file_path.stat().st_size, 'depth_range': 'Pending full parse'},
        'depth_analysis': {'start_depth': 'Pending', 'end_depth': 'Pending', 'step_size': 'Pending', 'total_samples': 0},
        'logs_data': [],
        'log_names': [],
        'comparison': {},
        'is_preview': True,
    }
    try:
        _norm_path = normalize_las_file(file_path)
        las = lasio.read(str(_norm_path), ignore_header_errors=True)
        if _norm_path != file_path and _norm_path.exists():
            try: _norm_path.unlink()
            except Exception: pass
        quick_info = {}
        for key, label in WELL_INFO_FIELDS:
            try:
                item = las.well.get(key)
                val = getattr(item, 'value', None) if item is not None else None
                if val not in [None, '', 'nan']:
                    quick_info[label] = val
            except Exception:
                pass
        if quick_info:
            preview['well_info'].update(quick_info)
            preview['well_name'] = str(quick_info.get('Well Name') or preview['well_name'])
            preview['summary']['well_name'] = preview['well_name']
        curve_names = []
        for c in getattr(las, 'curves', []) or []:
            m = str(getattr(c, 'mnemonic', '') or '').strip().upper()
            if m and m != 'DEPTH':
                curve_names.append(m)
        preview['total_logs'] = len(curve_names)
        preview['summary']['total_curves'] = len(curve_names)
        preview['available_logs'] = [{'mnemonic': m, 'unit': 'N/A', 'description': 'Quick preview'} for m in curve_names[:20]]
        preview['log_names'] = curve_names[:20]
        try:
            idx = list(getattr(las, 'index', []) or [])
            if idx:
                start = float(idx[0]); end = float(idx[-1])
                preview['depth_analysis']['start_depth'] = start
                preview['depth_analysis']['end_depth'] = end
                preview['summary']['depth_range'] = f"{start} - {end}"
        except Exception:
            pass
    except Exception:
        pass
    return to_builtin(preview)


def parse_las_file(file_path: Path):
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError('Empty file.')
    _norm_path = normalize_las_file(file_path)
    try:
        las = lasio.read(str(_norm_path), ignore_header_errors=False)
    except Exception as exc:
        if _norm_path != file_path and _norm_path.exists():
            try: _norm_path.unlink()
            except Exception: pass
        raise ValueError(f'Corrupted LAS file or unsupported structure: {exc}')
    if not getattr(las, 'curves', None):
        if _norm_path != file_path and _norm_path.exists():
            try: _norm_path.unlink()
            except Exception: pass
        raise ValueError('Missing headers or curves section in LAS file.')

    try:
        df = las.df().reset_index()
    except Exception as exc:
        if _norm_path != file_path and _norm_path.exists():
            try: _norm_path.unlink()
            except Exception: pass
        raise ValueError(f'Unable to convert LAS to tabular data: {exc}')
    if _norm_path != file_path and _norm_path.exists():
        try: _norm_path.unlink()
        except Exception: pass
    if df is None or df.empty:
        raise ValueError('LAS file contains no usable log samples.')

    depth_name = str(df.columns[0])
    df.rename(columns={depth_name: 'DEPTH'}, inplace=True)
    df.columns = [str(c).upper() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)

    null_value = None
    try:
        null_value = safe_float(getattr(getattr(las, 'well', {}), 'NULL').value)
    except Exception:
        pass
    # Replace the declared null value and common LAS null sentinel values
    null_values_to_replace = set()
    if null_value is not None:
        null_values_to_replace.add(null_value)
    # Common LAS null/absent data sentinels
    for common_null in [-999.25, -9999.25, -9999.0, -9998.0, 999.25, 9999.25, 9999.0]:
        null_values_to_replace.add(common_null)
    df = df.replace(list(null_values_to_replace), np.nan)
    # Also replace extreme outliers that are clearly null markers (> 5 std from mean)
    for col in df.columns:
        if col == 'DEPTH':
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        if s.notna().sum() > 5:
            mean, std = s.mean(), s.std()
            if std > 0:
                df[col] = s.where((s - mean).abs() <= 15 * std, other=np.nan)

    depth = pd.to_numeric(df['DEPTH'], errors='coerce')
    df['DEPTH'] = depth
    df = df.dropna(subset=['DEPTH']).reset_index(drop=True)
    if df.empty:
        raise ValueError('Depth index is missing or invalid.')

    curve_map = {str(c.mnemonic).upper(): c for c in las.curves}
    total_rows = len(df)
    depth_min = safe_float(df['DEPTH'].min())
    depth_max = safe_float(df['DEPTH'].max())
    depth_step = safe_float(df['DEPTH'].diff().replace(0, np.nan).median())

    # ── Extract depth unit from LAS file ──────────────────────────────────────
    # Priority: STRT header unit > first curve (DEPT/DEPTH) unit > fallback 'm'
    def _extract_depth_unit(las_obj):
        # Try well header STRT unit
        try:
            u = str(getattr(las_obj.well, 'STRT').unit).strip().upper()
            if u:
                return 'ft' if u in ('F', 'FT', 'FEET', 'FOOT') else 'm'
        except Exception:
            pass
        # Try first curve unit (usually DEPT or DEPTH)
        try:
            u = str(las_obj.curves[0].unit).strip().upper()
            if u:
                return 'ft' if u in ('F', 'FT', 'FEET', 'FOOT') else 'm'
        except Exception:
            pass
        return 'm'  # LAS 2.0 default

    depth_unit = _extract_depth_unit(las)

    available_logs = []
    stats_map = {}
    numeric_cols = []
    total_missing = 0
    total_nulls = 0
    total_valid = 0

    for idx, col in enumerate(df.columns):
        if col == 'DEPTH':
            continue
        series = pd.to_numeric(df[col], errors='coerce')
        valid = int(series.notna().sum())
        missing = int(series.isna().sum())
        total_missing += missing
        total_valid += valid
        curve = curve_map.get(col)
        descr = first_non_empty(getattr(curve, 'descr', None), 'N/A')
        unit = first_non_empty(getattr(curve, 'unit', None), 'N/A')
        mn = safe_float(series.min())
        mx = safe_float(series.max())
        if valid > 0:
            q = series.dropna()
            mode_vals = q.mode()
            mode_val = safe_float(mode_vals.iloc[0]) if not mode_vals.empty else None
            stats = {
                'minimum': mn,
                'maximum': mx,
                'mean': safe_float(q.mean()),
                'median': safe_float(q.median()),
                'mode': mode_val,
                'std': safe_float(q.std()),
                'variance': safe_float(q.var()),
                'p10': safe_float(q.quantile(0.10)),
                'p25': safe_float(q.quantile(0.25)),
                'p50': safe_float(q.quantile(0.50)),
                'p75': safe_float(q.quantile(0.75)),
                'p90': safe_float(q.quantile(0.90)),
                'missing_count': missing,
                'null_count': missing,
                'valid_samples': valid,
                'availability_pct': round((valid / total_rows) * 100, 2) if total_rows else 0,
                'depth_start': depth_min,
                'depth_end': depth_max,
                'unit': unit,
                'description': descr,
            }
            numeric_cols.append(col)
        else:
            stats = {
                'minimum': None, 'maximum': None, 'mean': None, 'median': None, 'mode': None,
                'std': None, 'variance': None, 'p10': None, 'p25': None, 'p50': None, 'p75': None,
                'p90': None, 'missing_count': missing, 'null_count': missing, 'valid_samples': 0,
                'availability_pct': 0.0, 'depth_start': depth_min, 'depth_end': depth_max,
                'unit': unit, 'description': descr,
            }
        available_logs.append({
            'mnemonic': col,
            'description': descr,
            'unit': unit,
            'curve_index': idx,
            'depth_coverage': f"{depth_min if depth_min is not None else 'N/A'} - {depth_max if depth_max is not None else 'N/A'}",
            'availability_pct': stats['availability_pct']
        })
        stats_map[col] = stats

    matrix = df[numeric_cols].apply(pd.to_numeric, errors='coerce') if numeric_cols else pd.DataFrame()
    corr = matrix.corr().fillna(0) if not matrix.empty else pd.DataFrame()
    comparison = []
    cols = list(matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            cval = safe_float(corr.loc[cols[i], cols[j]]) if not corr.empty else None
            if cval is not None:
                comparison.append({'x': cols[i], 'y': cols[j], 'correlation': round(cval, 4)})

    well_map = {}
    for key, label in WELL_INFO_FIELDS:
        value = 'N/A'
        try:
            item = getattr(las.well, key)
            value = first_non_empty(getattr(item, 'value', None), 'N/A')
        except Exception:
            if key == 'VERS':
                try:
                    item = getattr(las.version, key)
                    value = first_non_empty(getattr(item, 'value', None), 'N/A')
                except Exception:
                    value = 'N/A'
        well_map[label] = value

    well_name = first_non_empty(well_map.get('Well Name'), file_path.stem)
    summary = {
        'total_curves': len(available_logs),
        'total_samples': total_rows,
        'valid_records': total_valid,
        'missing_values': total_missing,
        'null_values': total_nulls,
        'file_size': file_path.stat().st_size,
        'depth_range': f"{depth_min if depth_min is not None else 'N/A'} - {depth_max if depth_max is not None else 'N/A'} ({depth_unit})",
        'well_name': well_name,
        'depth_unit': depth_unit,
    }
    depth_analysis = {
        'start_depth': depth_min,
        'end_depth': depth_max,
        'step_size': depth_step,
        'total_samples': total_rows,
        'depth_unit': depth_unit,
    }

    return to_builtin({
        'well_info': well_map,
        'available_logs': available_logs,
        'stats': stats_map,
        'summary': summary,
        'depth_analysis': depth_analysis,
        'logs_data': df.fillna(value=np.nan).replace({np.nan: None}).to_dict(orient='records'),
        'log_names': [x['mnemonic'] for x in available_logs],
        'comparison': comparison,
    })



def process_las_in_background(task_id, save_path, original_filename, user_email):
    save_path = Path(save_path)
    try:
        preview = build_quick_las_preview(save_path, original_filename, user_email)
        preview_id = preview['id']
        save_analysis_item(preview)
        update_task(task_id, status='processing', progress=25, message='Quick well preview ready. Full LAS parsing in progress...', analysis_id=preview_id)
        analysis = parse_las_file(save_path)
        item = {
            'id': preview_id,
            'file_name': original_filename,
            'upload_time': preview['upload_time'],
            'analysis_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'user_email': user_email,
            'well_name': analysis['summary']['well_name'],
            'total_logs': analysis['summary']['total_curves'],
            'file_path': str(save_path),
            **analysis,
            'is_preview': False,
        }
        save_analysis_item(item)
        update_task(task_id, status='completed', progress=100, message='LAS file processed successfully.', analysis_id=preview_id)
    except Exception as exc:
        update_task(task_id, status='failed', progress=0, message=str(exc))

@app.route('/')
def index():
    return redirect(url_for('dashboard'))


@app.post('/register')
def register():
    data = load_users()
    payload = request.get_json(silent=True) or request.form
    email = payload.get('email', '').strip().lower()
    password = payload.get('password', '').strip()
    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password are required.'}), 400
    if any(u['email'] == email for u in data['users']):
        return jsonify({'success': False, 'message': 'User already exists.'}), 400
    data['users'].append({'email': email, 'password': generate_password_hash(password)})
    save_users(data)
    session['user_email'] = email
    return jsonify({'success': True})


@app.post('/login')
def login():
    data = load_users()
    payload = request.get_json(silent=True) or request.form
    email = payload.get('email', '').strip().lower()
    password = payload.get('password', '').strip()
    for user in data['users']:
        if user['email'] == email and check_password_hash(user['password'], password):
            session['user_email'] = email
            return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials.'}), 401


@app.get('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/visualization')
@login_required
def visualization_page():
    return render_template('visualization.html')


@app.route('/prediction')
@login_required
def prediction_page():
    return render_template('prediction.html')


@app.route('/uncertainty')
@login_required
def uncertainty_page():
    return render_template('case_study.html')


@app.route('/export')
@login_required
def export_page():
    return render_template('export.html')


@app.post('/upload')
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Please choose a LAS file.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid LAS format. Supported formats: .las, .LAS'}), 400

    filename = secure_filename(file.filename)
    save_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
    save_path = UPLOAD_DIR / save_name
    file.save(save_path)

    if not save_path.exists() or save_path.stat().st_size == 0:
        save_path.unlink(missing_ok=True)
        return jsonify({'success': False, 'message': 'Empty file.'}), 400

    task_id = str(uuid.uuid4())
    set_task(task_id, {
        'status': 'queued',
        'progress': 5,
        'message': 'File uploaded. Waiting to start parsing...',
        'file_name': filename,
        'user_email': session.get('user_email'),
        'analysis_id': None
    })

    thread = threading.Thread(
        target=process_las_in_background,
        args=(task_id, str(save_path), filename, session.get('user_email')),
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'File uploaded successfully. Quick preview started, then full LAS processing will continue.'
    })

@app.get('/upload-status/<task_id>')
@login_required
def upload_status(task_id):
    task = get_task(task_id)
    if not task:
        return jsonify({'success': False, 'message': 'Task not found.'}), 404

    if task.get('user_email') != session.get('user_email'):
        return jsonify({'success': False, 'message': 'Unauthorized task access.'}), 403

    if task.get('status') == 'completed' and task.get('analysis_id'):
        store = load_history_store()
        for item in store.get('items', []):
            if item.get('id') == task['analysis_id']:
                session['analysis_id'] = item['id']
                session['las_path'] = item.get('file_path')
                return jsonify({
                    'success': True,
                    'status': 'completed',
                    'progress': task.get('progress', 100),
                    'message': task.get('message', ''),
                    'analysis': to_builtin({
                        'well_info': item['well_info'],
                        'available_logs': item['available_logs'],
                        'stats': item['stats'],
                        'summary': item['summary'],
                        'depth_analysis': item['depth_analysis'],
                        'log_names': item['log_names'],
                        'comparison': item['comparison'],
                    })
                })

    return jsonify({
        'success': True,
        'status': task.get('status', 'queued'),
        'progress': task.get('progress', 0),
        'message': task.get('message', 'Processing...')
    })


@app.get('/parse')
@login_required
def parse_endpoint():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No LAS file available.'}), 404
    return jsonify({'success': True, 'analysis': to_builtin({
        'well_info': item['well_info'],
        'available_logs': item['available_logs'],
        'stats': item['stats'],
        'summary': item['summary'],
        'depth_analysis': item['depth_analysis'],
        'log_names': item['log_names'],
        'comparison': item['comparison'],
    })})


@app.get('/well-info')
@login_required
def well_info():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    return jsonify({'success': True, 'well_info': to_builtin(item['well_info']), 'summary': to_builtin(item['summary']), 'depth_analysis': to_builtin(item['depth_analysis'])})


@app.get('/logs')
@login_required
def logs():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    selected = request.args.get('selected', '').strip()
    selected_logs = [x.strip().upper() for x in selected.split(',') if x.strip()] if selected else item['log_names']
    selected_logs = [x for x in selected_logs if x in item['log_names']]
    rows = []
    for record in item['logs_data']:
        row = {'DEPTH': record.get('DEPTH')}
        for key in selected_logs:
            row[key] = record.get(key)
        rows.append(row)
    return jsonify({'success': True, 'selected_logs': to_builtin(selected_logs), 'records': to_builtin(rows), 'available_logs': to_builtin(item['available_logs']), 'stats': to_builtin({k: v for k, v in item['stats'].items() if k in selected_logs})})




@app.get('/task-status/<task_id>')
@login_required
def task_status(task_id):
    task = get_task(task_id)
    if not task:
        return jsonify({'success': False, 'message': 'Task not found.'}), 404
    return jsonify({'success': True, **to_builtin(task)})

@app.get('/analysis-detail')
@login_required
def analysis_detail():
    try:
        item = load_current_analysis()
        if not item:
            return jsonify({'success': False, 'message': 'No LAS file loaded. Please upload a LAS file from the Dashboard first.'})
        # Exclude logs_data from response - it can be 100k+ rows
        safe_item = {k: v for k, v in item.items() if k != 'logs_data'}
        return jsonify({'success': True, 'analysis': to_builtin(safe_item)})
    except Exception as exc:
        return jsonify({'success': False, 'message': f'Analysis detail error: {str(exc)}'}), 500


@app.get('/uncertainty-logs')
@login_required
def uncertainty_logs():
    try:
        item = load_current_analysis()
        if not item:
            return jsonify({'success': False, 'message': 'No LAS file loaded.'})
        log_names = item.get('log_names', [])
        all_logs = [l for l in log_names if str(l).upper() != 'DEPTH']
        POROSITY_KEYS = {
            'NPHI','NPHISS','NPHIS','NPL','TNPH','TNPHI','CNPOR','CNC','NPOR','CNCF',
            'PHIN','BPHI','NPHI_STA','NPHI_D',
            'RHOB','RHOZ','DEN','ZDEN','RHOG',
            'DT','DTC','AC','DTCO','DTP','SONIC',
            'PHIT','PHIE','PHI','POROSITY','CPOR','DPHI',
        }
        SATURATION_KEYS = {
            'RT','RESD','RDEP','RDEEP','ILD','LLD','AT90','AHT90','HDRS','RILD',
            'RLLD','RLA5','RLA4','RLA3','RLA2','RLA1','MSFL','RXO','RSHAL',
            'ILM','LLM','RMED','RSHA','LL8','LLS','ILS',
            'SW','SWT','SWE','SWI',
        }
        def classify(name):
            n = str(name).upper()
            for k in POROSITY_KEYS:
                if n == k or n.startswith(k) or (len(k)>=3 and k in n):
                    return 'porosity'
            for k in SATURATION_KEYS:
                if n == k or n.startswith(k) or (len(k)>=3 and k in n):
                    return 'saturation'
            ltype = detect_log_type(n)
            if ltype in ('neutron','density','sonic'):
                return 'porosity'
            if ltype == 'resistivity':
                return 'saturation'
            return 'other'
        porosity_logs = [l for l in all_logs if classify(l) == 'porosity']
        saturation_logs = [l for l in all_logs if classify(l) == 'saturation']
        def best(lst, preferred):
            for p in preferred:
                m = next((l for l in lst if l.upper()==p.upper()), None)
                if m: return m
            return lst[0] if lst else None
        default_phi = best(porosity_logs, ['NPHISS','NPHIS','NPHI','RHOB','DT','PHIT','PHIE'])
        default_sw  = best(saturation_logs, ['ILD','RESD','RT','LLD','ILM','LL8'])
        return jsonify({
            'success': True,
            'porosity_logs': porosity_logs,
            'saturation_logs': saturation_logs,
            'all_logs': all_logs,
            'default_phi_log': default_phi,
            'default_sw_log': default_sw,
            'depth_min': item.get('start_depth'),
            'depth_max': item.get('end_depth'),
            'depth_unit': item.get('depth_analysis', {}).get('depth_unit', 'm'),
        })
    except Exception as exc:
        return jsonify({'success': False, 'message': str(exc)}), 500



@app.get('/current-analysis')
@login_required
def current_analysis():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No current analysis found.'}), 404
    return jsonify({
        'success': True,
        'analysis_id': item.get('id'),
        'file_name': item.get('file_name'),
        'well_info': to_builtin(item.get('well_info', {})),
        'summary': to_builtin(item.get('summary', {})),
        'available_logs': to_builtin([{**log, 'is_resistivity': ('RES' in str(log.get('mnemonic','')).upper() or 'RT' in str(log.get('mnemonic','')).upper() or 'ILD' in str(log.get('mnemonic','')).upper() or 'LLD' in str(log.get('mnemonic','')).upper())} for log in item.get('available_logs', [])]),
        'log_names': to_builtin(item.get('log_names', [])),
        'stats': to_builtin(item.get('stats', {})),
        'depth_analysis': to_builtin(item.get('depth_analysis', {}))
    })



@app.get('/prediction-preview')
@login_required
def prediction_preview():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    bundle = build_prediction_bundle(item)
    return jsonify({'success': True, 'rows': bundle['preview'][:5]})




@app.get('/export-predictions/<kind>')
@login_required
def export_predictions(kind):
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    bundle = build_prediction_bundle(item)
    kind = kind.lower()
    mapping = {
        'vsh': ('vsh', 'drakeai_vsh_predictions.csv'),
        'porosity': ('porosity', 'drakeai_porosity_predictions.csv'),
        'saturation': ('saturation', 'drakeai_water_saturation_predictions.csv'),
        'lithology': ('lithology', 'drakeai_lithology_predictions.csv')
    }
    sections = compute_prediction_sections(item)
    if kind not in mapping:
        return jsonify({'success': False, 'message': 'Invalid prediction export type.'}), 400
    key, filename = mapping[kind]
    if sections.get('success') and key in sections.get('exports', {}):
        df = pd.DataFrame(sections['exports'][key])
    else:
        df = pd.DataFrame(bundle.get(key, []))
    mem = io.BytesIO(df.to_csv(index=False).encode('utf-8'))
    mem.seek(0)
    response = send_file(mem, mimetype='text/csv', as_attachment=True, download_name=filename, conditional=True)
    response.headers['Cache-Control'] = 'no-store'
    return response






@app.get('/prediction-store-status')
@login_required
def prediction_store_status():
    """Check if AI prediction data exists in the backend by doing a quick computation."""
    item = load_current_analysis()
    if not item:
        return jsonify({'success': True, 'has_data': False, 'store': {}})
    try:
        import numpy as np
        df = pd.DataFrame(item.get('logs_data', []))
        if df.empty:
            return jsonify({'success': True, 'has_data': False, 'store': {}})
        df.columns = [str(c).upper() for c in df.columns]

        sects = compute_prediction_sections(item)
        por_rows = sects.get('exports', {}).get('porosity', []) if sects.get('success') else []
        sat_rows = sects.get('exports', {}).get('saturation', []) if sects.get('success') else []

        por_valid = [r for r in por_rows if r.get('PHIT') is not None]
        sat_valid = [r for r in sat_rows if r.get('SW') is not None]

        has_data = len(por_valid) > 0 or len(sat_valid) > 0

        store = {
            'porosity_count': len(por_valid),
            'saturation_count': len(sat_valid),
        }
        return jsonify({'success': True, 'has_data': has_data, 'store': store})
    except Exception as exc:
        return jsonify({'success': True, 'has_data': False, 'store': {}, 'error': str(exc)})


@app.post('/prediction-sections')
@login_required
def prediction_sections():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(compute_prediction_sections(item, payload))
    except Exception as exc:
        return jsonify({'success': False, 'message': f'Prediction calculation failed: {str(exc)}'}), 500

@app.get('/export-report/full-zip')
@login_required
def export_full_zip():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404

    # Use real compute_prediction_sections (same engine as AI Prediction tab)
    sections = compute_prediction_sections(item)
    exports  = sections.get('exports', {})
    vsh_df   = pd.DataFrame(exports.get('vsh',        []))
    por_df   = pd.DataFrame(exports.get('porosity',   []))
    sat_df   = pd.DataFrame(exports.get('saturation', []))
    lith_df  = pd.DataFrame(exports.get('lithology',  []))

    logs_df   = pd.DataFrame(item.get('logs_data', []))
    stats     = to_builtin(item.get('stats', {}))
    well_name = item.get('well_name', 'Unknown Well')

    # Build combined CSV
    def safe_cols(df, cols):
        return df[[c for c in cols if c in df.columns]] if not df.empty else pd.DataFrame()

    if not vsh_df.empty:
        combined = vsh_df.copy()
        for src, cols in [
            (por_df,  ['PHIT', 'PHIE']),
            (sat_df,  ['RT', 'SW', 'SATURATION_METHOD']),
            (lith_df, ['RHOB', 'LITHOLOGY'])
        ]:
            if not src.empty:
                combined = combined.merge(safe_cols(src, ['DEPTH'] + cols), on='DEPTH', how='left')
    else:
        combined = por_df if not por_df.empty else sat_df if not sat_df.empty else lith_df

    # Plotly graph HTML builder
    PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
    dark_bg = "rgba(13,26,42,1)"
    grid_c  = "rgba(255,255,255,0.08)"

    def make_graph_html(title, traces_json, layout_json):
        body = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>" + title + "</title>"
            "<script src='" + PLOTLY_CDN + "'></script>"
            "<style>body{background:" + dark_bg + ";color:#dceaf4;font-family:sans-serif;margin:0;padding:16px;}"
            "h2{color:#9dc8e8;}p{color:#7a9bbf;font-size:.9rem;}</style>"
            "</head><body>"
            "<h2>" + title + "</h2>"
            "<p>Well: " + well_name + "</p>"
            "<div id='c' style='width:100%;height:85vh;'></div>"
            "<script>Plotly.newPlot('c'," + traces_json + "," + layout_json + ",{responsive:true,displaylogo:false});</script>"
            "</body></html>"
        )
        return body

    base_layout = {
        "paper_bgcolor": dark_bg, "plot_bgcolor": dark_bg,
        "font": {"color": "#dceaf4", "size": 11},
        "margin": {"l": 70, "r": 30, "t": 60, "b": 50},
        "showlegend": True
    }

    def yax():
        return {"title": "Depth (m)", "autorange": "reversed",
                "gridcolor": grid_c, "zeroline": False}

    def xax(title):
        return {"title": title, "gridcolor": grid_c, "zeroline": False}

    def line_trace(x, y, name, color, xaxis="x", yaxis="y"):
        return {"x": x, "y": y, "type": "scattergl", "mode": "lines", "name": name,
                "line": {"color": color, "width": 1.5}, "connectgaps": True,
                "xaxis": xaxis, "yaxis": yaxis,
                "hovertemplate": "<b>Depth:</b> %{y:.2f}<br><b>" + name + ":</b> %{x:.4f}<extra></extra>"}

    def dcol(df):
        return df['DEPTH'].tolist() if 'DEPTH' in df.columns and not df.empty else []

    # 1. VSH graph
    vsh_traces = []
    if not vsh_df.empty:
        d = dcol(vsh_df)
        if 'GR'  in vsh_df.columns: vsh_traces.append(line_trace(vsh_df['GR'].tolist(),  d, 'GR (API)', '#31d17c'))
        if 'IGR' in vsh_df.columns: vsh_traces.append(line_trace(vsh_df['IGR'].tolist(), d, 'IGR',      '#f5c542'))
        if 'VSH' in vsh_df.columns: vsh_traces.append(line_trace(vsh_df['VSH'].tolist(), d, 'VSH',      '#2f80ff'))
    vsh_layout = {**base_layout, "title": "VSH Prediction — " + well_name,
                  "yaxis": yax(), "xaxis": xax("GR / IGR / VSH")}
    html_vsh = make_graph_html("VSH — " + well_name, json.dumps(vsh_traces), json.dumps(vsh_layout))

    # 2. Porosity graph
    por_traces = []
    if not por_df.empty:
        d = dcol(por_df)
        if 'PHIT' in por_df.columns: por_traces.append(line_trace(por_df['PHIT'].tolist(), d, 'PHIT', '#00d1ff'))
        if 'PHIE' in por_df.columns: por_traces.append(line_trace(por_df['PHIE'].tolist(), d, 'PHIE', '#c792ff'))
    por_layout = {**base_layout, "title": "Porosity — " + well_name,
                  "yaxis": yax(), "xaxis": xax("Porosity (fraction)")}
    html_por = make_graph_html("Porosity — " + well_name, json.dumps(por_traces), json.dumps(por_layout))

    # 3. Saturation graph
    sat_traces = []
    if not sat_df.empty:
        d = dcol(sat_df)
        if 'SW' in sat_df.columns: sat_traces.append(line_trace(sat_df['SW'].tolist(), d, 'SW',        '#f5c542'))
        if 'RT' in sat_df.columns: sat_traces.append(line_trace(sat_df['RT'].tolist(), d, 'RT (ohm.m)','#ff7a90'))
    sat_layout = {**base_layout, "title": "Water Saturation — " + well_name,
                  "yaxis": yax(), "xaxis": xax("Sw (fraction) / RT (ohm.m)")}
    html_sat = make_graph_html("Water Saturation — " + well_name, json.dumps(sat_traces), json.dumps(sat_layout))

    # 4. Lithology classification bar graph
    lith_colors = {
        'Clean Sandstone': '#2f80ff', 'Shaly Sand': '#9dc8e8', 'Shale': '#dba96e',
        'Limestone': '#d4d47a', 'Dolomite': '#c08ade', 'Coal': '#888888', 'Unknown': '#555555'
    }
    lith_traces = []
    if not lith_df.empty and 'LITHOLOGY' in lith_df.columns and 'DEPTH' in lith_df.columns:
        for lname, grp in lith_df.groupby('LITHOLOGY'):
            lith_traces.append({
                "x": [lname] * len(grp), "y": grp['DEPTH'].tolist(),
                "type": "bar", "name": lname, "orientation": "v",
                "marker": {"color": lith_colors.get(lname, '#9ec3ff')},
                "hovertemplate": "<b>" + lname + "</b><br>Depth: %{y:.2f}<extra></extra>"
            })
    lith_layout = {**base_layout, "title": "Lithology Classification — " + well_name,
                   "yaxis": yax(), "xaxis": xax("Lithology"), "barmode": "overlay"}
    html_lith = make_graph_html("Lithology — " + well_name, json.dumps(lith_traces), json.dumps(lith_layout))

    # 5. Combined 4-panel summary
    combo_traces = []
    combo_layout = {
        **base_layout,
        "title": "Combined Reservoir Summary — " + well_name,
        "yaxis":  {**yax(), "domain": [0, 1]},
        "yaxis2": {"matches": "y", "showticklabels": False, "autorange": "reversed", "gridcolor": grid_c, "zeroline": False},
        "yaxis3": {"matches": "y", "showticklabels": False, "autorange": "reversed", "gridcolor": grid_c, "zeroline": False},
        "yaxis4": {"matches": "y", "showticklabels": False, "autorange": "reversed", "gridcolor": grid_c, "zeroline": False},
        "xaxis":  {**xax("VSH"),        "domain": [0.00, 0.22]},
        "xaxis2": {**xax("PHIT"),       "domain": [0.26, 0.48]},
        "xaxis3": {**xax("SW"),         "domain": [0.52, 0.74]},
        "xaxis4": {**xax("RHOB g/cc"),  "domain": [0.78, 1.00]},
    }
    if not vsh_df.empty  and 'VSH'  in vsh_df.columns:
        combo_traces.append(line_trace(vsh_df['VSH'].tolist(),   dcol(vsh_df),  'VSH',  '#2f80ff', 'x',  'y'))
    if not por_df.empty  and 'PHIT' in por_df.columns:
        combo_traces.append(line_trace(por_df['PHIT'].tolist(),  dcol(por_df),  'PHIT', '#00d1ff', 'x2', 'y2'))
    if not sat_df.empty  and 'SW'   in sat_df.columns:
        combo_traces.append(line_trace(sat_df['SW'].tolist(),    dcol(sat_df),  'SW',   '#f5c542', 'x3', 'y3'))
    if not lith_df.empty and 'RHOB' in lith_df.columns:
        combo_traces.append(line_trace(lith_df['RHOB'].tolist(), dcol(lith_df), 'RHOB', '#ff7a90', 'x4', 'y4'))
    html_combined = make_graph_html("Combined Summary — " + well_name, json.dumps(combo_traces), json.dumps(combo_layout))

    # Build ZIP
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('well_info.json',        json.dumps(to_builtin(item.get('well_info', {})), indent=2))
        zf.writestr('analysis_summary.json', json.dumps(to_builtin(item.get('summary', {})), indent=2))
        zf.writestr('log_properties.json',   json.dumps(stats, indent=2))
        zf.writestr('well_log_values.csv',   logs_df.to_csv(index=False))
        zf.writestr('predictions/vsh_predictions.csv',              vsh_df.to_csv(index=False))
        zf.writestr('predictions/porosity_predictions.csv',         por_df.to_csv(index=False))
        zf.writestr('predictions/water_saturation_predictions.csv', sat_df.to_csv(index=False))
        zf.writestr('predictions/lithology_predictions.csv',        lith_df.to_csv(index=False))
        zf.writestr('predictions/all_reservoir_parameters.csv',     combined.to_csv(index=False))
        zf.writestr('graphs/vsh_graph.html',              html_vsh)
        zf.writestr('graphs/porosity_graph.html',         html_por)
        zf.writestr('graphs/saturation_graph.html',       html_sat)
        zf.writestr('graphs/lithology_graph.html',        html_lith)
        zf.writestr('graphs/combined_summary_graph.html', html_combined)
    mem.seek(0)
    response = send_file(mem, mimetype='application/zip', as_attachment=True, download_name='drakeai_full_results.zip')
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/analysis-history')

@login_required
def analysis_history():
    store = load_history_store()
    items = to_builtin([
        {
            'id': x['id'],
            'file_name': x.get('file_name'),
            'upload_time': x.get('upload_time'),
            'analysis_date': x.get('analysis_date'),
            'well_name': x.get('well_name'),
            'total_logs': x.get('total_logs'),
        }
        for x in store.get('items', []) if x.get('user_email') == session.get('user_email')
    ])
    return jsonify({'success': True, 'items': to_builtin(items)})


@app.post('/analysis-history/<analysis_id>/load')
@login_required
def load_history_item(analysis_id):
    store = load_history_store()
    for item in store.get('items', []):
        if item.get('id') == analysis_id and item.get('user_email') == session.get('user_email'):
            session['analysis_id'] = analysis_id
            session['las_path'] = item.get('file_path')
            return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'History item not found.'}), 404


@app.delete('/analysis-history/<analysis_id>')
@login_required
def delete_history_item(analysis_id):
    store = load_history_store()
    before = len(store.get('items', []))
    store['items'] = [x for x in store.get('items', []) if not (x.get('id') == analysis_id and x.get('user_email') == session.get('user_email'))]
    if len(store['items']) == before:
        return jsonify({'success': False, 'message': 'History item not found.'}), 404
    save_history_store(store)
    if session.get('analysis_id') == analysis_id:
        session.pop('analysis_id', None)
        session.pop('las_path', None)
    return jsonify({'success': True})


@app.get('/compare-logs')
@login_required
def compare_logs():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    selected = request.args.get('selected', '').strip()
    selected_logs = [x.strip().upper() for x in selected.split(',') if x.strip()] if selected else item['log_names'][:4]
    selected_logs = [x for x in selected_logs if x in item['log_names']]
    data = pd.DataFrame(item['logs_data'])
    cols = ['DEPTH'] + selected_logs
    data = data[cols].copy()
    corr = data[selected_logs].corr().fillna(0) if selected_logs else pd.DataFrame()
    pairs = []
    for i in range(len(selected_logs)):
        for j in range(i + 1, len(selected_logs)):
            sample = data[['DEPTH', selected_logs[i], selected_logs[j]]].dropna().head(400)
            pairs.append({
                'x': selected_logs[i],
                'y': selected_logs[j],
                'correlation': round(float(corr.loc[selected_logs[i], selected_logs[j]]), 4) if not corr.empty else None,
                'points': to_builtin(sample.to_dict(orient='records'))
            })
    return jsonify({'success': True, 'matrix': to_builtin(corr.to_dict()) if not corr.empty else {}, 'pairs': to_builtin(pairs)})



@app.post('/predict-porosity')
@login_required
def predict_porosity():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    bundle = build_prediction_bundle(item)
    return jsonify({'success': True, 'data': bundle['porosity']})



@app.post('/predict-saturation')
@login_required
def predict_saturation():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    bundle = build_prediction_bundle(item)
    return jsonify({'success': True, 'data': bundle['saturation']})



@app.post('/predict-lithology')
@login_required
def predict_lithology():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    bundle = build_prediction_bundle(item)
    return jsonify({'success': True, 'data': bundle['lithology']})


@app.get('/export/<fmt>')

@login_required
def export_data(fmt):
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    fmt = fmt.lower()
    summary_df = pd.DataFrame([item['summary']])
    well_df = pd.DataFrame([item['well_info']])
    stats_df = pd.DataFrame([{ 'mnemonic': k, **v } for k, v in item['stats'].items()])

    if fmt == 'csv':
        csv_buffer = io.StringIO()
        stats_df.to_csv(csv_buffer, index=False)
        mem = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        mem.seek(0)
        response = send_file(mem, mimetype='text/csv', as_attachment=True, download_name='drakeai_log_statistics.csv', conditional=True)
        response.headers['Cache-Control'] = 'no-store'
        return response
    if fmt == 'json':
        mem = io.BytesIO(json.dumps(to_builtin({'well_info': item['well_info'], 'summary': item['summary'], 'stats': item['stats']}), indent=2).encode('utf-8'))
        mem.seek(0)
        response = send_file(mem, mimetype='application/json', as_attachment=True, download_name='drakeai_analysis.json', conditional=True)
        response.headers['Cache-Control'] = 'no-store'
        return response
    if fmt == 'excel':
        mem = io.BytesIO()
        with pd.ExcelWriter(mem, engine='openpyxl') as writer:
            well_df.to_excel(writer, sheet_name='WellInfo', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            stats_df.to_excel(writer, sheet_name='LogStatistics', index=False)
        mem.seek(0)
        return send_file(mem, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='drakeai_analysis.xlsx')
    if fmt == 'pdf':
        content = [
            'DrakeAI LAS Analysis Report', '', 'Well Information:'
        ]
        content += [f"{k}: {v}" for k, v in item['well_info'].items()]
        content += ['', 'Summary:']
        content += [f"{k}: {v}" for k, v in item['summary'].items()]
        content += ['', 'Log Statistics:']
        for row in stats_df.to_dict(orient='records')[:50]:
            content.append(f"{row['mnemonic']} | min={row.get('minimum')} max={row.get('maximum')} mean={row.get('mean')} unit={row.get('unit')}")
        mem = io.BytesIO('\n'.join(content).encode('utf-8'))
        mem.seek(0)
        response = send_file(mem, mimetype='application/pdf', as_attachment=True, download_name='drakeai_report.pdf', conditional=True)
        response.headers['Cache-Control'] = 'no-store'
        return response
    return jsonify({'success': False, 'message': 'Unsupported export format.'}), 400


# ─────────────────────────────────────────────────────────────────────────────
# LAS LOG VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

LOG_TYPE_MAP = {
    # Gamma Ray — all common field naming conventions
    'GR': 'gamma_ray', 'GRD': 'gamma_ray', 'GRS': 'gamma_ray', 'GRR': 'gamma_ray',
    'CGR': 'gamma_ray', 'SGR': 'gamma_ray', 'HSGR': 'gamma_ray', 'GRC': 'gamma_ray',
    'GAMMA': 'gamma_ray', 'GAMMARAY': 'gamma_ray', 'GR_RAW': 'gamma_ray',
    # Resistivity
    'RT': 'resistivity', 'RESD': 'resistivity', 'RDEP': 'resistivity', 'RDEEP': 'resistivity',
    'ILD': 'resistivity', 'LLD': 'resistivity', 'AT90': 'resistivity',
    'AHT90': 'resistivity', 'HDRS': 'resistivity', 'RILD': 'resistivity',
    'RLLD': 'resistivity', 'RLA5': 'resistivity', 'RLA4': 'resistivity',
    'RLA3': 'resistivity', 'RLA2': 'resistivity', 'RLA1': 'resistivity',
    'MSFL': 'resistivity', 'RXO': 'resistivity', 'RSHAL': 'resistivity',
    'ILM': 'resistivity', 'LLM': 'resistivity', 'RMED': 'resistivity', 'RSHA': 'resistivity',
    # Density
    'RHOB': 'density', 'RHOZ': 'density', 'DEN': 'density', 'ZDEN': 'density',
    'RHOG': 'density', 'DRHOB': 'density', 'RHOZ_STA': 'density',
    # Neutron — ALL common naming variants
    'NPHI': 'neutron', 'NPHISS': 'neutron', 'NPHIS': 'neutron', 'NPL': 'neutron',
    'TNPH': 'neutron', 'TNPHI': 'neutron', 'CNPOR': 'neutron', 'CNC': 'neutron',
    'NPOR': 'neutron', 'CNCF': 'neutron', 'PHIN': 'neutron', 'BPHI': 'neutron',
    'NPHI_STA': 'neutron', 'NPHI_D': 'neutron',
    # Sonic
    'DT': 'sonic', 'DTC': 'sonic', 'AC': 'sonic', 'DTCO': 'sonic',
    'DTP': 'sonic', 'DTSM': 'sonic', 'DTL': 'sonic', 'SONIC': 'sonic',
    # Caliper
    'CALI': 'caliper', 'CAL': 'caliper', 'HCAL': 'caliper',
    # SP
    'SP': 'sp',
    # PE
    'PE': 'pe', 'PEF': 'pe', 'PEFZ': 'pe',
}

STANDARD_SCALES = {
    'gamma_ray':  {'unit': 'API',     'scale': 'linear',  'xmin': 0,    'xmax': 200,  'reverse': False},
    'resistivity':{'unit': 'ohm.m',   'scale': 'log',     'xmin': 0.2,  'xmax': 2000, 'reverse': False},
    'density':    {'unit': 'g/cc',    'scale': 'linear',  'xmin': 1.8,  'xmax': 2.8,  'reverse': False},
    'neutron':    {'unit': 'fraction','scale': 'linear',  'xmin': 0.54, 'xmax': 0,    'reverse': True},
    'sonic':      {'unit': 'us/ft',   'scale': 'linear',  'xmin': 40,   'xmax': 160,  'reverse': False},
    'caliper':    {'unit': 'inch',    'scale': 'linear',  'xmin': 6,    'xmax': 16,   'reverse': False},
    'sp':         {'unit': 'mV',      'scale': 'linear',  'xmin': None, 'xmax': None, 'reverse': False},
    'pe':         {'unit': 'b/e',     'scale': 'linear',  'xmin': 0,    'xmax': 10,   'reverse': False},
    'unknown':    {'unit': '',        'scale': 'linear',  'xmin': None, 'xmax': None, 'reverse': False},
}


def detect_log_type(curve_name, unit='', description=''):
    """Detect log type using exact → prefix → substring → unit/description matching."""
    n = str(curve_name).upper().strip()
    # 1. Exact match
    if n in LOG_TYPE_MAP:
        return LOG_TYPE_MAP[n]
    # 2. Curve name starts with a known key (e.g. NPHISS starts with NPHI)
    for key, ltype in LOG_TYPE_MAP.items():
        if n.startswith(key) and n != 'DEPTH':
            return ltype
    # 3. Known key starts with curve name (e.g. curve=NPL matched by NPHI key)
    for key, ltype in LOG_TYPE_MAP.items():
        if key.startswith(n) and len(n) >= 2 and n != 'DEPTH':
            return ltype
    # 4. Substring: key is inside curve name (e.g. NPHISS contains NPHI)
    for key, ltype in LOG_TYPE_MAP.items():
        if len(key) >= 3 and key in n and n != 'DEPTH':
            return ltype
    # 5. Unit-based fallback
    u = str(unit).lower()
    d = str(description).lower()
    if 'api' in u or 'gapi' in u or 'gamma' in d:
        return 'gamma_ray'
    if 'ohm' in u or 'resist' in d:
        return 'resistivity'
    if 'g/cc' in u or 'g/c3' in u or 'density' in d or 'bulk' in d:
        return 'density'
    if 'frac' in u or 'dec' in u or 'neutron' in d or ('porosity' in d and 'sonic' not in d):
        return 'neutron'
    if 'us/ft' in u or 'us/m' in u or 'usec' in u or 'sonic' in d or 'transit' in d:
        return 'sonic'
    if ('in' in u and 'inch' in u) or 'caliper' in d:
        return 'caliper'
    if 'mv' in u or 'spontaneous' in d:
        return 'sp'
    return 'unknown'


def get_standard_scale(log_type):
    return STANDARD_SCALES.get(log_type, STANDARD_SCALES['unknown']).copy()


def calculate_log_summary_data(item):
    df = pd.DataFrame(item.get('logs_data', []))
    if df.empty:
        return []
    log_names = item.get('log_names', [])
    stats = item.get('stats', {})
    rows = []
    for col in log_names:
        if col not in df.columns:
            continue
        s = stats.get(col, {})
        series = pd.to_numeric(df[col], errors='coerce')
        log_type = detect_log_type(col, s.get('unit', ''), s.get('description', ''))
        scale_info = get_standard_scale(log_type)
        rows.append({
            'curve': col,
            'description': s.get('description', 'N/A'),
            'unit_las': s.get('unit', 'N/A'),
            'min': safe_float(s.get('minimum')),
            'max': safe_float(s.get('maximum')),
            'mean': safe_float(s.get('mean')),
            'median': safe_float(s.get('median')),
            'std': safe_float(s.get('std')),
            'null_count': s.get('null_count', 0),
            'valid_count': s.get('valid_samples', 0),
            'log_type': log_type.replace('_', ' ').title(),
            'suggested_unit': scale_info['unit'],
            'suggested_xmin': scale_info['xmin'],
            'suggested_xmax': scale_info['xmax'],
            'suggested_scale': scale_info['scale'],
        })
    return to_builtin(rows)


def generate_ai_log_interpretation(log_summary, log_names):
    upper_names = [str(n).upper() for n in log_names]
    detected = [r['curve'] for r in log_summary]
    gr_logs  = [r['curve'] for r in log_summary if r['log_type'].lower() == 'gamma ray']
    res_logs = [r['curve'] for r in log_summary if r['log_type'].lower() == 'resistivity']
    den_logs = [r['curve'] for r in log_summary if r['log_type'].lower() == 'density']
    neu_logs = [r['curve'] for r in log_summary if r['log_type'].lower() == 'neutron']
    son_logs = [r['curve'] for r in log_summary if r['log_type'].lower() == 'sonic']
    msgs = []
    if gr_logs:
        msgs.append(f"✅ Gamma Ray logs detected: {', '.join(gr_logs)}. Suitable for Vsh calculation and shale volume estimation.")
        # check if GR values suggest clean/shaly
        for r in log_summary:
            if r['curve'] in gr_logs and r['mean'] is not None:
                if r['mean'] < 60:
                    msgs.append(f"📊 {r['curve']} mean ({r['mean']:.1f} API) suggests predominantly clean intervals, but requires local calibration.")
                elif r['mean'] > 100:
                    msgs.append(f"📊 {r['curve']} mean ({r['mean']:.1f} API) suggests shaly intervals. Vsh will be elevated. Requires calibration.")
                else:
                    msgs.append(f"📊 {r['curve']} mean ({r['mean']:.1f} API) suggests mixed sand-shale intervals.")
    else:
        msgs.append("⚠️ No Gamma Ray logs detected. Vsh calculation will not be available unless a custom GR log is selected.")
    if res_logs:
        msgs.append(f"✅ Resistivity logs detected: {', '.join(res_logs)}. Suitable for water saturation (Archie/Indonesia) calculation.")
        for r in log_summary:
            if r['curve'] in res_logs and r['max'] is not None and r['max'] > 20:
                msgs.append(f"📊 {r['curve']} shows high values (max={r['max']:.1f} ohm.m). Possible reservoir-quality interval or hydrocarbon presence — requires calibration with core/test/formation water data for confirmation.")
                break
    else:
        msgs.append("⚠️ No resistivity logs detected. Water saturation calculation requires a resistivity log.")
    if den_logs and neu_logs:
        msgs.append(f"✅ Density ({', '.join(den_logs)}) and Neutron ({', '.join(neu_logs)}) logs available. Suitable for porosity calculation and crossplot lithology analysis.")
    elif den_logs:
        msgs.append(f"✅ Density logs detected: {', '.join(den_logs)}. Suitable for density porosity calculation.")
    elif neu_logs:
        msgs.append(f"✅ Neutron porosity logs detected: {', '.join(neu_logs)}. Suitable for neutron porosity calculation.")
    else:
        msgs.append("⚠️ No density or neutron porosity logs detected. Porosity calculation options will be limited.")
    if son_logs:
        msgs.append(f"✅ Sonic/DT logs detected: {', '.join(son_logs)}. Suitable for sonic porosity (Wyllie) and acoustic impedance / rock stiffness interpretation.")
    missing_standard = []
    for name, aliases in [('GR',['GR','GRD','CGR']),('Resistivity',['RT','RESD','ILD','LLD']),('RHOB',['RHOB','RHOZ','DEN']),('NPHI',['NPHI','TNPH','CNC']),('DT',['DT','DTC','DTCO'])]:
        if not any(a in upper_names for a in aliases):
            missing_standard.append(name)
    if missing_standard:
        msgs.append(f"⚠️ Standard logs missing from this LAS file: {', '.join(missing_standard)}.")
    workflow_logs = []
    if gr_logs: workflow_logs.append("Vsh (GR-based)")
    if den_logs or neu_logs or son_logs: workflow_logs.append("Porosity (RHOB/NPHI/DT)")
    if res_logs: workflow_logs.append("Saturation (Archie/Indonesia)")
    if gr_logs and (den_logs or neu_logs): workflow_logs.append("Lithology classification")
    if workflow_logs:
        msgs.append(f"🔬 Available petrophysical workflows: {', '.join(workflow_logs)}.")
    msgs.append("📌 Note: All interpretations require calibration with core data, formation water salinity (Rw), pressure data, and/or interpreted pay intervals before use in reservoir models.")
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY QUANTIFICATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_porosity_uncertainty(phi_p50, method='fixed', uncertainty_value=0.03,
                                   pct=0.10, measured=None):
    phi_p50 = np.array(phi_p50, dtype=float)
    if method == 'residual' and measured is not None:
        measured = np.array(measured, dtype=float)
        residuals = measured - phi_p50
        sigma = float(np.nanstd(residuals))
        spread = 1.2816 * sigma  # scalar
    elif method == 'percent':
        spread = phi_p50 * float(pct)  # per-depth array
    else:
        # 'fixed' mode: use a per-depth spread that scales with value
        # Base spread = uncertainty_value, but vary it by ±30% relative to local value
        # so curves are not flat. Minimum spread = 0.5 * uncertainty_value.
        base = float(uncertainty_value)
        phi_safe = np.where(np.isnan(phi_p50), 0.15, phi_p50)
        phi_mean = float(np.nanmean(phi_safe)) if np.nanmean(phi_safe) > 0 else 0.15
        # spread varies proportionally to local phi deviation from mean
        deviation = np.abs(phi_safe - phi_mean) / (phi_mean + 1e-6)
        spread = base * (1.0 + deviation)  # per-depth array
    phi_p10 = np.clip(phi_p50 - spread, 0, 1)
    phi_p90 = np.clip(phi_p50 + spread, 0, 1)
    return phi_p10, phi_p50, phi_p90


def calculate_saturation_uncertainty(sw_p50, method='fixed', uncertainty_value=0.05,
                                     pct=0.10, measured=None):
    sw_p50 = np.array(sw_p50, dtype=float)
    if method == 'residual' and measured is not None:
        measured = np.array(measured, dtype=float)
        residuals = measured - sw_p50
        sigma = float(np.nanstd(residuals))
        spread = 1.2816 * sigma  # scalar
    elif method == 'percent':
        spread = sw_p50 * float(pct)  # per-depth array
    else:
        # 'fixed' mode: vary spread per-depth proportionally to local sw deviation
        base = float(uncertainty_value)
        sw_safe = np.where(np.isnan(sw_p50), 0.5, sw_p50)
        sw_mean = float(np.nanmean(sw_safe)) if np.nanmean(sw_safe) > 0 else 0.5
        deviation = np.abs(sw_safe - sw_mean) / (sw_mean + 1e-6)
        spread = base * (1.0 + deviation)  # per-depth array
    sw_p10 = np.clip(sw_p50 - spread, 0, 1)
    sw_p90 = np.clip(sw_p50 + spread, 0, 1)
    return sw_p10, sw_p50, sw_p90


def interpret_uncertainty_results(p10_arr, p50_arr, p90_arr, kind='porosity'):
    spreads = np.array(p90_arr) - np.array(p10_arr)
    mean_spread = float(np.nanmean(spreads))
    max_spread_idx = int(np.nanargmax(spreads))
    min_spread_idx = int(np.nanargmin(spreads))
    msgs = []
    if kind == 'porosity':
        msgs.append(f"📊 Average porosity uncertainty spread (P90-P10): {mean_spread:.4f} fraction.")
        msgs.append(f"🎯 Highest uncertainty near index {max_spread_idx}. These zones may benefit from core/calibrated log validation.")
        msgs.append(f"✅ Lowest uncertainty near index {min_spread_idx}. These are higher-confidence porosity prediction zones.")
        msgs.append("📌 P50 is the best-estimate curve. P10 and P90 are probabilistic uncertainty bounds, not separate correct answers.")
        msgs.append("📌 Accuracy depends on calibration with measured core porosity, NMR porosity, or pressure data.")
        if mean_spread < 0.03:
            msgs.append("✅ Narrow P10-P90 band detected: high-confidence porosity estimate across most of the interval.")
        elif mean_spread > 0.08:
            msgs.append("⚠️ Wide P10-P90 band detected: significant porosity uncertainty. Recommend additional data validation.")
    else:
        msgs.append(f"📊 Average Sw uncertainty spread (P90-P10): {mean_spread:.4f} fraction.")
        msgs.append(f"🎯 Highest Sw uncertainty near index {max_spread_idx}.")
        msgs.append("⚠️ Saturation uncertainty is largest in shaly zones, low-resistivity zones, poor porosity zones, and zones with uncertain Rw/Rsh/m/n values.")
        msgs.append("📌 P50 is the most likely Sw. P10 and P90 represent low/high uncertainty scenarios.")
        msgs.append("📌 Archie/Indonesia Sw uncertainty can be large due to sensitivity to Rw, Rt, PHIE, m, n, and Vsh.")
        if mean_spread < 0.05:
            msgs.append("✅ Narrow Sw spread: high confidence in saturation estimate.")
        elif mean_spread > 0.15:
            msgs.append("⚠️ Wide Sw spread: recommend reviewing Rw, m, n parameters and validating with core or test data.")
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# NEW API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/las-log-summary')
@login_required
def las_log_summary():
    try:
        item = load_current_analysis()
        if not item:
            return jsonify({'success': False, 'message': 'No LAS file loaded.'})
        summary = calculate_log_summary_data(item)
        log_names = item.get('log_names', [])
        interpretation = generate_ai_log_interpretation(summary, log_names)
        return jsonify({'success': True, 'summary': summary, 'interpretation': interpretation})
    except Exception as exc:
        return jsonify({'success': False, 'message': f'Summary error: {str(exc)}'}), 500


@app.get('/las-log-data')
@login_required
def las_log_data():
    try:
        item = load_current_analysis()
        if not item:
            return jsonify({'success': False, 'message': 'No LAS file loaded. Please upload a LAS file from the Dashboard.'})
        selected = request.args.get('curves', '').strip()
        log_names = item.get('log_names', [])
        if selected:
            requested = [x.strip() for x in selected.split(',') if x.strip()]
            resolved = []
            for req in requested:
                # Exact case-insensitive match first
                upper_map = {n.upper(): n for n in log_names}
                exact = upper_map.get(req.upper())
                if exact:
                    if exact not in resolved:
                        resolved.append(exact)
                else:
                    # Fuzzy match: find any log whose name contains or starts with req
                    match = find_log_name(log_names, [req])
                    if match and match not in resolved:
                        resolved.append(match)
            curves = resolved if resolved else log_names[:5]
        else:
            curves = log_names[:5]
        raw_records = item.get('logs_data', [])
        if not raw_records:
            return jsonify({'success': False, 'message': 'LAS file has no data records. Please re-upload the file.'})
        df_viz = pd.DataFrame(raw_records)
        # Make all columns uppercase to match log_names
        df_viz.columns = [str(c).upper() for c in df_viz.columns]
        cols_needed = ['DEPTH'] + [c for c in curves if c in df_viz.columns]
        if len(cols_needed) == 1:
            return jsonify({'success': False, 'message': f'None of the requested curves {curves} found in data. Available: {list(df_viz.columns)}'})
        df_viz = df_viz[cols_needed].copy()
        for col in cols_needed:
            df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce').astype('float64')
        # Replace NaN with None for JSON serialization
        rows = [{k: (None if (v is not None and isinstance(v, float) and math.isnan(v)) else v)
                 for k, v in row.items()}
                for row in df_viz.to_dict(orient='records')]
    except Exception as exc:
        return jsonify({'success': False, 'message': f'Data fetch error: {str(exc)}'}), 500
    stats = item.get('stats', {})
    curve_meta = []
    for c in curves:
        s = stats.get(c, {})
        log_type = detect_log_type(c, s.get('unit', ''), s.get('description', ''))
        scale = get_standard_scale(log_type)
        xmin = scale['xmin']
        xmax = scale['xmax']
        # auto-detect if no standard scale
        if xmin is None and s.get('minimum') is not None:
            xmin = s['minimum']
        if xmax is None and s.get('maximum') is not None:
            xmax = s['maximum']
        curve_meta.append({
            'curve': c,
            'unit': s.get('unit', scale['unit']),
            'log_type': log_type,
            'scale': scale['scale'],
            'xmin': xmin,
            'xmax': xmax,
            'reverse': scale['reverse'],
        })
    warnings_list = []
    for cm in curve_meta:
        c = cm['curve']
        s = stats.get(c, {})
        if cm['log_type'] == 'resistivity':
            mn = s.get('minimum')
            if mn is not None and float(mn) <= 0:
                warnings_list.append(f"{c}: Contains zero or negative resistivity values. These will be excluded from log-scale plotting.")
        if cm['log_type'] == 'neutron':
            mx = s.get('maximum')
            if mx is not None and float(mx) > 1:
                warnings_list.append(f"{c}: Values appear to be in percent (max={mx}). Consider converting to fraction by dividing by 100.")
        if cm['log_type'] == 'gamma_ray':
            mx = s.get('maximum')
            if mx is not None and float(mx) > 200:
                warnings_list.append(f"{c}: GR values exceed 200 API standard scale maximum (max={mx}). Displayed on 0-200 scale with clipping warning.")
        if cm['log_type'] == 'density':
            mn = s.get('minimum')
            mx = s.get('maximum')
            if (mn is not None and float(mn) < 1.8) or (mx is not None and float(mx) > 2.8):
                warnings_list.append(f"{c}: RHOB values outside standard 1.8-2.8 g/cc range. Standard scale applied but out-of-range values exist.")
    return jsonify({
        'success': True,
        'records': to_builtin(rows),
        'curve_meta': to_builtin(curve_meta),
        'warnings': warnings_list,
        'available_logs': to_builtin(item.get('available_logs', [])),
    })


@app.get('/download-log-summary-csv')
@login_required
def download_log_summary_csv():
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    summary = calculate_log_summary_data(item)
    df = pd.DataFrame(summary)
    mem = io.BytesIO(df.to_csv(index=False).encode('utf-8'))
    mem.seek(0)
    resp = send_file(mem, mimetype='text/csv', as_attachment=True,
                     download_name='drakeai_log_ranges_summary.csv', conditional=True)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.post('/compute-uncertainty')
@login_required
def compute_uncertainty():
    try:
        item = load_current_analysis()
        if not item:
            return jsonify({'success': False, 'message': 'No LAS file loaded. Please upload a LAS file from the Dashboard.'})
        payload = request.get_json(silent=True) or {}
        df = pd.DataFrame(item.get('logs_data', []))
        if df.empty:
            return jsonify({'success': False, 'message': 'No log data found in the active LAS file.'})
        # Uppercase all column names for consistency
        df.columns = [str(c).upper() for c in df.columns]
        if 'DEPTH' not in df.columns:
            df['DEPTH'] = np.arange(len(df), dtype=float)

        # ── POROSITY UNCERTAINTY ──
        phi_col = payload.get('phi_col', '')
        phi_method = payload.get('phi_method', 'percent')   # default percent for varying spread
        phi_unc = float(payload.get('phi_unc', 0.03))
        phi_pct = float(payload.get('phi_pct', 0.10))
        phi_meas_col = payload.get('phi_meas_col', '')

        # Always compute prediction sections first to get real PHIT/SW values
        _pred_sects = None
        try:
            _pred_sects = compute_prediction_sections(item)
        except Exception:
            pass

        phi_src = None
        # 1) Use computed PHIT from prediction sections — actual per-depth calculated values
        if _pred_sects and _pred_sects.get('success'):
            por_rows_pred = _pred_sects.get('exports', {}).get('porosity', [])
            if por_rows_pred:
                phit_vals = [r.get('PHIT') for r in por_rows_pred]
                candidate = pd.Series(phit_vals, dtype=float)
                if not candidate.isna().all():
                    phi_src = candidate
        # 2) Fall back to explicit phi_col from LAS if no prediction data
        if (phi_src is None or phi_src.isna().all()) and phi_col and phi_col.upper() in {c.upper() for c in df.columns}:
            real_col = next(c for c in df.columns if c.upper() == phi_col.upper())
            candidate = pd.to_numeric(df[real_col], errors='coerce')
            if not candidate.isna().all():
                phi_src = candidate
        # 3) Try common porosity column names directly from LAS
        if phi_src is None or phi_src.isna().all():
            for try_col in ['PHIT','PHIE','PHI','NPHI','NPHISS','NPHIS','POROSITY','CPOR','DPHI']:
                match = next((c for c in df.columns if c.upper()==try_col), None)
                if match:
                    candidate = pd.to_numeric(df[match], errors='coerce')
                    if not candidate.isna().all():
                        phi_src = candidate
                        break
        if phi_src is None or phi_src.isna().all():
            # final fallback
            phi_src = pd.Series([0.15] * len(df), dtype=float)

        phi_measured = None
        if phi_meas_col and phi_meas_col.upper() in {c.upper() for c in df.columns}:
            real_pm = next(c for c in df.columns if c.upper() == phi_meas_col.upper())
            if phi_method == 'residual':
                phi_measured = pd.to_numeric(df[real_pm], errors='coerce').values

        p10_phi, p50_phi, p90_phi = calculate_porosity_uncertainty(
            phi_src.fillna(phi_src.median()).values,
            method=phi_method, uncertainty_value=phi_unc, pct=phi_pct, measured=phi_measured
        )

        # ── SATURATION UNCERTAINTY ──
        sw_col = payload.get('sw_col', '')
        sw_method = payload.get('sw_method', 'percent')     # default percent for varying spread
        sw_unc = float(payload.get('sw_unc', 0.05))
        sw_pct = float(payload.get('sw_pct', 0.10))
        sw_meas_col = payload.get('sw_meas_col', '')

        sw_src = None
        # 1) Use computed SW from prediction sections — actual per-depth calculated values
        if _pred_sects and _pred_sects.get('success'):
            sat_rows_pred = _pred_sects.get('exports', {}).get('saturation', [])
            if sat_rows_pred:
                sw_vals = [r.get('SW') for r in sat_rows_pred]
                candidate_sw_pred = pd.Series(sw_vals, dtype=float)
                if not candidate_sw_pred.isna().all():
                    sw_src = candidate_sw_pred
        # 2) Fall back to explicit sw_col from LAS if no prediction data
        if (sw_src is None or sw_src.isna().all()) and sw_col and sw_col.upper() in {c.upper() for c in df.columns}:
            real_sw_col = next(c for c in df.columns if c.upper() == sw_col.upper())
            candidate_sw = pd.to_numeric(df[real_sw_col], errors='coerce')
            if not candidate_sw.isna().all():
                sw_src = candidate_sw
        # 3) Try common Sw column names directly from LAS
        if sw_src is None or sw_src.isna().all():
            for try_col in ['SW','SWT','SWE','WATER_SAT','SWI','SW_ARCHIE']:
                match = next((c for c in df.columns if c.upper()==try_col), None)
                if match:
                    candidate = pd.to_numeric(df[match], errors='coerce')
                    if not candidate.isna().all():
                        sw_src = candidate
                        break
        if sw_src is None or sw_src.isna().all():
            sw_src = pd.Series([0.50] * len(df), dtype=float)

        sw_measured = None
        if sw_meas_col and sw_meas_col.upper() in {c.upper() for c in df.columns}:
            real_sm = next(c for c in df.columns if c.upper() == sw_meas_col.upper())
            if sw_method == 'residual':
                sw_measured = pd.to_numeric(df[real_sm], errors='coerce').values

        p10_sw, p50_sw, p90_sw = calculate_saturation_uncertainty(
            sw_src.fillna(sw_src.median()).values,
            method=sw_method, uncertainty_value=sw_unc, pct=sw_pct, measured=sw_measured
        )

        depth_raw = pd.to_numeric(df['DEPTH'], errors='coerce').values

        # Align all arrays to the same length using the raw depth as master
        n_raw = len(depth_raw)

        def _align(arr, n):
            arr = np.array(arr, dtype=float)
            if len(arr) >= n:
                return arr[:n]
            # pad with last value if shorter (e.g. from prediction sections)
            pad = np.full(n - len(arr), arr[-1] if len(arr) > 0 else np.nan)
            return np.concatenate([arr, pad])

        p10_phi = _align(p10_phi, n_raw)
        p50_phi = _align(p50_phi, n_raw)
        p90_phi = _align(p90_phi, n_raw)
        p10_sw  = _align(p10_sw,  n_raw)
        p50_sw  = _align(p50_sw,  n_raw)
        p90_sw  = _align(p90_sw,  n_raw)

        phi_spread = p90_phi - p10_phi
        sw_spread  = p90_sw  - p10_sw

        # Build raw records
        raw_records = []
        for i in range(n_raw):
            d = depth_raw[i]
            if d is None or np.isnan(float(d) if d is not None else float('nan')):
                continue
            raw_records.append({
                'DEPTH': float(d),
                'PHI_P10': float(p10_phi[i]), 'PHI_P50': float(p50_phi[i]), 'PHI_P90': float(p90_phi[i]),
                'PHI_UNCERTAINTY_SPREAD': float(phi_spread[i]),
                'SW_P10':  float(p10_sw[i]),  'SW_P50':  float(p50_sw[i]),  'SW_P90':  float(p90_sw[i]),
                'SW_UNCERTAINTY_SPREAD':  float(sw_spread[i]),
            })

        # Sort by depth ascending so lines render correctly (no zigzag)
        raw_records.sort(key=lambda r: r['DEPTH'])

        n = len(raw_records)

        # Round for JSON output
        records = []
        for r in raw_records:
            records.append({
                'DEPTH': to_builtin(round(r['DEPTH'], 2)),
                'PHI_P10': to_builtin(round(r['PHI_P10'], 5)),
                'PHI_P50': to_builtin(round(r['PHI_P50'], 5)),
                'PHI_P90': to_builtin(round(r['PHI_P90'], 5)),
                'PHI_UNCERTAINTY_SPREAD': to_builtin(round(r['PHI_UNCERTAINTY_SPREAD'], 5)),
                'SW_P10':  to_builtin(round(r['SW_P10'],  5)),
                'SW_P50':  to_builtin(round(r['SW_P50'],  5)),
                'SW_P90':  to_builtin(round(r['SW_P90'],  5)),
                'SW_UNCERTAINTY_SPREAD':  to_builtin(round(r['SW_UNCERTAINTY_SPREAD'],  5)),
            })

        # Extract sorted depth/value arrays from final records for stats
        sorted_depths   = [r['DEPTH']   for r in raw_records]
        sorted_phi_p50  = [r['PHI_P50'] for r in raw_records]
        sorted_phi_p10  = [r['PHI_P10'] for r in raw_records]
        sorted_phi_p90  = [r['PHI_P90'] for r in raw_records]
        sorted_sw_p50   = [r['SW_P50']  for r in raw_records]
        sorted_sw_p10   = [r['SW_P10']  for r in raw_records]
        sorted_sw_p90   = [r['SW_P90']  for r in raw_records]
        sorted_phi_spread = [r['PHI_UNCERTAINTY_SPREAD'] for r in raw_records]
        sorted_sw_spread  = [r['SW_UNCERTAINTY_SPREAD']  for r in raw_records]

        phi_interp = interpret_uncertainty_results(sorted_phi_p10, sorted_phi_p50, sorted_phi_p90, 'porosity')
        sw_interp  = interpret_uncertainty_results(sorted_sw_p10,  sorted_sw_p50,  sorted_sw_p90,  'saturation')

        avg_phi_p50    = float(np.nanmean(sorted_phi_p50))   if sorted_phi_p50  else 0.0
        avg_phi_spread = float(np.nanmean(sorted_phi_spread)) if sorted_phi_spread else 0.0
        avg_sw_p50     = float(np.nanmean(sorted_sw_p50))    if sorted_sw_p50   else 0.0
        avg_sw_spread  = float(np.nanmean(sorted_sw_spread))  if sorted_sw_spread else 0.0

        max_phi_depth = sorted_depths[int(np.nanargmax(sorted_phi_spread))] if sorted_phi_spread else None
        max_sw_depth  = sorted_depths[int(np.nanargmax(sorted_sw_spread))]  if sorted_sw_spread  else None

        return jsonify({
            'success': True,
            'records': to_builtin([r for r in records if r.get('PHI_P50') is not None and r.get('DEPTH') is not None][:5]),
            'all_records': to_builtin(records),
            'phi_interp': phi_interp,
            'sw_interp': sw_interp,
            'summary_cards': {
                'avg_phi_p50':   round(avg_phi_p50,    4),
                'avg_phi_spread':round(avg_phi_spread,  4),
                'avg_sw_p50':    round(avg_sw_p50,     4),
                'avg_sw_spread': round(avg_sw_spread,   4),
                'max_phi_spread_depth': to_builtin(max_phi_depth),
                'max_sw_spread_depth':  to_builtin(max_sw_depth),
            }
        })
    except Exception as exc:
        import traceback
        return jsonify({'success': False, 'message': f'Uncertainty computation error: {str(exc)}'}), 500


@app.get('/download-uncertainty-csv/<kind>')
@login_required
def download_uncertainty_csv(kind):
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis found.'}), 404
    df_raw = pd.DataFrame(item.get('logs_data', []))
    if df_raw.empty:
        return jsonify({'success': False, 'message': 'No log data.'}), 404
    df_raw.columns = [str(c).upper() for c in df_raw.columns]
    if 'DEPTH' not in df_raw.columns:
        df_raw['DEPTH'] = np.arange(len(df_raw), dtype=float)

    sects = compute_prediction_sections(item)
    por_rows = sects.get('exports', {}).get('porosity', []) if sects.get('success') else []
    sat_rows = sects.get('exports', {}).get('saturation', []) if sects.get('success') else []

    # ── Use prediction export rows directly (already depth-ordered, same count as LAS) ──
    # This avoids float-depth matching issues entirely.
    if por_rows:
        por_df_exp = pd.DataFrame(por_rows)
        por_df_exp.columns = [str(c).upper() for c in por_df_exp.columns]
        phi_depth = pd.to_numeric(por_df_exp.get('DEPTH', pd.Series(dtype=float)), errors='coerce').values
        phi_values = pd.to_numeric(por_df_exp.get('PHIT', pd.Series(dtype=float)), errors='coerce').values
    else:
        phi_depth = pd.to_numeric(df_raw['DEPTH'], errors='coerce').values
        phi_values = np.full(len(phi_depth), np.nan)
        # Fallback to raw LAS porosity column
        for try_col in ['PHIT','PHIE','PHI','NPHI','NPHISS','NPHIS','POROSITY','CPOR','DPHI']:
            match = next((c for c in df_raw.columns if c.upper() == try_col), None)
            if match:
                phi_values = pd.to_numeric(df_raw[match], errors='coerce').values
                break

    if sat_rows:
        sat_df_exp = pd.DataFrame(sat_rows)
        sat_df_exp.columns = [str(c).upper() for c in sat_df_exp.columns]
        sw_depth = pd.to_numeric(sat_df_exp.get('DEPTH', pd.Series(dtype=float)), errors='coerce').values
        sw_values = pd.to_numeric(sat_df_exp.get('SW', pd.Series(dtype=float)), errors='coerce').values
    else:
        sw_depth = pd.to_numeric(df_raw['DEPTH'], errors='coerce').values
        sw_values = np.full(len(sw_depth), np.nan)
        for try_col in ['SW','SWT','SWE','WATER_SAT','SWI','SW_ARCHIE']:
            match = next((c for c in df_raw.columns if c.upper() == try_col), None)
            if match:
                sw_values = pd.to_numeric(df_raw[match], errors='coerce').values
                break

    # Fill NaN with median so uncertainty functions receive clean arrays
    phi_med = float(np.nanmedian(phi_values)) if not np.all(np.isnan(phi_values)) else 0.15
    sw_med  = float(np.nanmedian(sw_values))  if not np.all(np.isnan(sw_values))  else 0.50
    phi_clean = np.where(np.isnan(phi_values), phi_med, phi_values)
    sw_clean  = np.where(np.isnan(sw_values),  sw_med,  sw_values)

    # Compute uncertainty using percent method so spread varies per depth
    p10_phi, p50_phi, p90_phi = calculate_porosity_uncertainty(
        phi_clean, method='percent', pct=0.10)
    p10_sw, p50_sw, p90_sw = calculate_saturation_uncertainty(
        sw_clean, method='percent', pct=0.10)

    if kind == 'porosity':
        depth_out = phi_depth
        mask = ~np.isnan(depth_out.astype(float))
        df_out = pd.DataFrame({
            'DEPTH':   depth_out[mask],
            'PHI_P10': p10_phi[mask], 'PHI_P50': p50_phi[mask], 'PHI_P90': p90_phi[mask],
            'PHI_UNCERTAINTY_SPREAD': (p90_phi - p10_phi)[mask]
        })
        fname = 'drakeai_porosity_uncertainty.csv'
    else:
        depth_out = sw_depth
        mask = ~np.isnan(depth_out.astype(float))
        df_out = pd.DataFrame({
            'DEPTH':  depth_out[mask],
            'SW_P10': p10_sw[mask], 'SW_P50': p50_sw[mask], 'SW_P90': p90_sw[mask],
            'SW_UNCERTAINTY_SPREAD': (p90_sw - p10_sw)[mask]
        })
        fname = 'drakeai_saturation_uncertainty.csv'

    df_out = df_out.round(6)
    mem = io.BytesIO(df_out.to_csv(index=False).encode('utf-8'))
    mem.seek(0)
    resp = send_file(mem, mimetype='text/csv', as_attachment=True, download_name=fname, conditional=True)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


def _pred_cache_path(user_email, analysis_id):
    """Return a safe file path for storing prediction results for a given user/analysis."""
    safe_user = (user_email or 'anon').replace('@', '_at_').replace('.', '_').replace('/', '_')
    safe_aid  = (str(analysis_id) or 'none').replace('/', '_').replace('\\', '_')
    return PRED_CACHE_DIR / f'{safe_user}__{safe_aid}__pred.json'


@app.post('/prediction-cache/save')
@login_required
def save_prediction_cache():
    """Save prediction results to server so they survive page navigation."""
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'message': 'No analysis loaded.'}), 400
    payload = request.get_json(silent=True) or {}
    user_email  = session.get('user_email', 'anon')
    analysis_id = item.get('id', 'unknown')
    cache_path  = _pred_cache_path(user_email, analysis_id)
    try:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        return jsonify({'success': True})
    except Exception as exc:
        return jsonify({'success': False, 'message': str(exc)}), 500


@app.get('/prediction-cache/load')
@login_required
def load_prediction_cache():
    """Load previously saved prediction results from server."""
    item = load_current_analysis()
    if not item:
        return jsonify({'success': False, 'has_data': False})
    user_email  = session.get('user_email', 'anon')
    analysis_id = item.get('id', 'unknown')
    cache_path  = _pred_cache_path(user_email, analysis_id)
    if not cache_path.exists():
        return jsonify({'success': True, 'has_data': False, 'store': {}})
    try:
        data = json.loads(cache_path.read_text(encoding='utf-8'))
        return jsonify({'success': True, 'has_data': True, 'store': data})
    except Exception as exc:
        return jsonify({'success': True, 'has_data': False, 'store': {}, 'error': str(exc)})


if __name__ == '__main__':
    app.run(debug=True)
