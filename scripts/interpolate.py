"""IDW delle osservazioni orarie su una griglia regolare, nel formato di leaflet-velocity.
Uso: python scripts/interpolate.py 2026-03-01  ->  data/wind-<giorno>.json (una lista di frame, uno per ora)"""
import json, sys
import numpy as np
from pyproj import Transformer

day = sys.argv[1]
obs = json.load(open(f"data/raw/obs-{day}.json"))
to_ll = Transformer.from_crs(32632, 4326, always_xy=True)  # coordinate UTM 32N verificate: ricadono in Sardegna
xy = np.array([to_ll.transform(s["est"], s["nord"]) for s in obs])

LON0, LON1, LAT0, LAT1, D = 8.0, 9.9, 38.8, 41.3, 0.05
nx, ny = round((LON1 - LON0) / D) + 1, round((LAT1 - LAT0) / D) + 1
gx, gy = np.meshgrid(LON0 + D * np.arange(nx), LAT1 - D * np.arange(ny))  # riga 0 = nord

def idw(vals, ok, p=2):
    d2 = (gx[..., None] - xy[ok, 0]) ** 2 + (gy[..., None] - xy[ok, 1]) ** 2
    w = 1 / np.maximum(d2, 1e-6) ** (p / 2)
    return (w * vals[ok]).sum(-1) / w.sum(-1)

frames = []
for h in range(24):
    t = f"{day}T{h:02d}:00:00"
    u = np.full(len(obs), np.nan); v = u.copy()
    for i, s in enumerate(obs):
        r = next((d for d in s["dati"] if d["t"] == t), None)
        if r:  # direzione meteorologica: da dove soffia
            u[i] = -r["vel"] * np.sin(np.radians(r["dir"])); v[i] = -r["vel"] * np.cos(np.radians(r["dir"]))
    ok = ~np.isnan(u)
    if ok.sum() < 5: continue  # troppo poche stazioni per interpolare
    hdr = dict(lo1=LON0, la1=LAT1, lo2=LON1, la2=LAT0, dx=D, dy=D, nx=nx, ny=ny, refTime=t, nstazioni=int(ok.sum()))
    frames.append([{"header": dict(hdr, parameterCategory=2, parameterNumber=2), "data": np.round(idw(u, ok), 2).ravel().tolist()},
                   {"header": dict(hdr, parameterCategory=2, parameterNumber=3), "data": np.round(idw(v, ok), 2).ravel().tolist()}])
json.dump(frames, open(f"data/wind-{day}.json", "w"))
assert frames and len(frames[0][0]["data"]) == nx * ny
print(f"{len(frames)} frame, griglia {nx}x{ny}")
