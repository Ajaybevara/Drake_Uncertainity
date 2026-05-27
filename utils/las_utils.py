import lasio
import pandas as pd
import numpy as np

ALIASES = {'DEPTH':['DEPT','DEPTH','MD'],'GR':['GR','GAMMA','GAM'],'RHOB':['RHOB','RHOZ','DEN'],'DT':['DT','DTC','SONIC'],'RT':['RT','ILD','LLD','RES','AT90'],'NPHI':['NPHI','NPH','TNPH'],'CALI':['CALI','HCAL']}

def parse_las_file(path):
    las = lasio.read(path)
    df = las.df().reset_index()
    df.rename(columns={df.columns[0]:'DEPTH'}, inplace=True)
    df.columns = [str(c).strip().upper() for c in df.columns]
    df.replace([-999.25,-999.0,-9999.0,-99999.0], np.nan, inplace=True)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors='coerce').interpolate(limit_direction='both')
    ren = {}
    for std, opts in ALIASES.items():
        for c in df.columns:
            if c.upper() in opts and c != std:
                ren[c] = std
                break
    if ren: df.rename(columns=ren, inplace=True)
    return las, df

def summarize_well_info(las, df):
    def gv(k, d='N/A'):
        try: return str(getattr(las.well, k).value)
        except: return d
    return {'WELL':gv('WELL'),'FIELD':gv('FLD'),'COMPANY':gv('COMP'),'LOCATION':gv('LOC'),'API':gv('API'),'START_DEPTH':f"{float(df['DEPTH'].min()):.2f}" if 'DEPTH' in df.columns else 'N/A','END_DEPTH':f"{float(df['DEPTH'].max()):.2f}" if 'DEPTH' in df.columns else 'N/A','STEP':f"{float(df['DEPTH'].diff().median()):.2f}" if len(df)>1 else 'N/A','CURVES':', '.join(df.columns.tolist())}

def get_available_logs(df): return [c for c in df.columns if c != 'DEPTH']

def curve_units_map(las):
    out = {}
    try:
        for c in las.curves:
            out[str(c.mnemonic).strip().upper()] = str(c.unit).strip()
    except Exception:
        pass
    return out
