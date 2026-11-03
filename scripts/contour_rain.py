"""Isoiete giornaliere: interpola i mm (RBF liscia) e ne ricava bande e linee in GeoJSON.
Uso: python scripts/contour_rain.py  ->  data/rain-iso.json  {giorno: {bande, linee}}"""
import json
import numpy as np
from contourpy import contour_generator, FillType, LineType
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

LIVELLI = [0.2, 1, 2, 5, 10, 20, 35, 50, 80]  # mm; devono coincidere con la legenda in index.html
MASK_KM = 15  # oltre questa distanza dalla stazione piu' vicina non si disegna nulla (mare, zone vuote)
LON0, LON1, LAT0, LAT1, D = 8.0, 9.9, 38.8, 41.3, 0.02

rain = json.load(open("data/rain.json"))
gx, gy = np.meshgrid(np.arange(LON0, LON1 + D / 2, D), np.arange(LAT0, LAT1 + D / 2, D))
km = lambda lon, lat: np.c_[np.asarray(lon) * 111.32 * np.cos(np.radians(40)), np.asarray(lat) * 110.57]  # equirettangolare, basta per 15 km
far = cKDTree(km(*zip(*[(s["lon"], s["lat"]) for s in rain["stazioni"]]))).query(km(gx.ravel(), gy.ravel()))[0].reshape(gx.shape) > MASK_KM

def ring(a): return [[round(float(x), 4), round(float(y), 4)] for x, y in a]
out = {}
for i, giorno in enumerate(rain["giorni"]):
    pts = [(s["lon"], s["lat"], s["mm"][i]) for s in rain["stazioni"] if s["mm"][i] is not None]
    xy, mm = np.array([p[:2] for p in pts]), np.array([p[2] for p in pts])
    # smoothing: attenua i picchi isolati (stazione singola) invece di fare "bolle"
    z = RBFInterpolator(km(xy[:, 0], xy[:, 1]), mm, kernel="thin_plate_spline", smoothing=1.0)(km(gx.ravel(), gy.ravel())).reshape(gx.shape)
    z = np.clip(z, 0, mm.max())  # la RBF puo' andare sotto 0 o oltre il massimo misurato
    g = contour_generator(gx, gy, np.ma.array(z, mask=far), fill_type=FillType.OuterOffset, line_type=LineType.Separate)
    bande = []
    for lo, hi in zip(LIVELLI, LIVELLI[1:] + [1e9]):
        polys, offs = g.filled(lo, hi)
        coords = [[ring(p[o[k]:o[k + 1]]) for k in range(len(o) - 1)] for p, o in zip(polys, offs)]
        if coords: bande.append({"type": "Feature", "properties": {"da": lo}, "geometry": {"type": "MultiPolygon", "coordinates": coords}})
    linee = [{"type": "Feature", "properties": {"mm": lv}, "geometry": {"type": "MultiLineString", "coordinates": [ring(l) for l in g.lines(lv)]}}
             for lv in LIVELLI if g.lines(lv)]
    out[giorno] = {"bande": {"type": "FeatureCollection", "features": bande}, "linee": {"type": "FeatureCollection", "features": linee}}
    print(giorno, "bande", len(bande), "livelli con linee", len(linee), "max stazioni", mm.max(), "max griglia", round(float(z[~far].max()), 1))
json.dump(out, open("data/rain-iso.json", "w"), separators=(",", ":"))
