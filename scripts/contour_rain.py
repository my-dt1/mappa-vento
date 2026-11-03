"""Isoiete giornaliere: interpola i mm (RBF), leviga, ricava bande e linee e le ritaglia sulla Sardegna.
Uso: python scripts/contour_rain.py  ->  data/rain-iso.json  {giorno: {bande, linee}}
Confine: data/sardegna.geojson (ISTAT, semplificato)."""
import json
import numpy as np
import shapely
from contourpy import contour_generator, FillType, LineType
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape

LIVELLI = [0.2, 1, 2, 5, 10, 20, 35, 50, 80]  # mm; devono coincidere con la legenda in index.html
MASK_KM = 15  # oltre questa distanza dalla stazione piu' vicina non si disegna nulla
LON0, LON1, LAT0, LAT1, D = 8.0, 9.9, 38.8, 41.4, 0.005  # ~500 m: curve lisce, non a scalini
SMOOTH = 2  # sigma del filtro gaussiano, in celle (~1 km)
TOL = 0.0003  # semplificazione finale (~30 m): riduce il peso del file senza spigoli visibili

terra = shapely.make_valid(shape(json.load(open("data/sardegna.geojson"))["geometry"]))
rain = json.load(open("data/rain.json"))
gx, gy = np.meshgrid(np.arange(LON0, LON1 + D / 2, D), np.arange(LAT0, LAT1 + D / 2, D))
km = lambda lon, lat: np.c_[np.asarray(lon) * 111.32 * np.cos(np.radians(40)), np.asarray(lat) * 110.57]  # equirettangolare, basta per 15 km
far = cKDTree(km([s["lon"] for s in rain["stazioni"]], [s["lat"] for s in rain["stazioni"]])).query(km(gx.ravel(), gy.ravel()))[0].reshape(gx.shape) > MASK_KM
grid_km = km(gx.ravel(), gy.ravel())

def clip(g):
    return shapely.set_precision(g.intersection(terra).simplify(TOL), 1e-4)

out = {}
for i, giorno in enumerate(rain["giorni"]):
    pts = [(s["lon"], s["lat"], s["mm"][i]) for s in rain["stazioni"] if s["mm"][i] is not None]
    xy, mm = np.array([p[:2] for p in pts]), np.array([p[2] for p in pts])
    # smoothing dell'RBF: attenua i picchi di una sola stazione; il gaussiano toglie i residui angolosi
    z = RBFInterpolator(km(xy[:, 0], xy[:, 1]), mm, kernel="thin_plate_spline", smoothing=1.0)(grid_km).reshape(gx.shape)
    z = np.clip(gaussian_filter(z, SMOOTH), 0, mm.max())  # la RBF puo' andare sotto 0 o oltre il massimo misurato
    g = contour_generator(gx, gy, np.ma.array(z, mask=far), fill_type=FillType.OuterOffset, line_type=LineType.Separate)
    bande, linee = [], []
    for lo, hi in zip(LIVELLI, LIVELLI[1:] + [1e9]):
        polys, offs = g.filled(lo, hi)
        parti = []
        for p, o in zip(polys, offs):  # primo anello = esterno, gli altri = buchi
            anelli = [p[o[k]:o[k + 1]] for k in range(len(o) - 1)]
            parti.append(Polygon(anelli[0], anelli[1:]))
        geom = clip(shapely.make_valid(MultiPolygon(parti))) if parti else None
        if geom is not None and not geom.is_empty:
            bande.append({"type": "Feature", "properties": {"da": lo}, "geometry": mapping(geom)})
    for lv in LIVELLI:
        tratti = [LineString(l) for l in g.lines(lv) if len(l) > 1]
        if not tratti: continue
        geom = clip(shapely.MultiLineString(tratti))
        if not geom.is_empty:
            linee.append({"type": "Feature", "properties": {"mm": lv}, "geometry": mapping(geom)})
    out[giorno] = {"bande": {"type": "FeatureCollection", "features": bande}, "linee": {"type": "FeatureCollection", "features": linee}}
    print(giorno, "bande", len(bande), "linee", len(linee), "max stazioni", mm.max(), "max griglia", round(float(z[~far].max()), 1))
json.dump(out, open("data/rain-iso.json", "w"), separators=(",", ":"))
