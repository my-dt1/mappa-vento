"""Pioggia giornaliera (PCG, mm) di tutte le stazioni attive in un intervallo di giorni.
Uso: WIND_API_KEY=... python scripts/fetch_rain.py 2026-03-01 2026-03-07  ->  data/rain.json"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pyproj import Transformer

B = "https://hmj6uzslsb.execute-api.eu-north-1.amazonaws.com/v1"
KEY = os.environ["WIND_API_KEY"]
da, a = sys.argv[1], sys.argv[2]
giorni = [(date.fromisoformat(da) + timedelta(n)).isoformat() for n in range((date.fromisoformat(a) - date.fromisoformat(da)).days + 1)]

def get(path, params):
    url = f"{B}/{path}?" + urllib.parse.urlencode(params)
    for i in range(6):  # l'API risponde 429/500 sotto carico
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"x-api-key": KEY}), timeout=60))
        except Exception as e:
            err = e; time.sleep(2 * (i + 1))
    raise err

def station(s):
    r = get("misure", {"stazione": s["cod_stazione"], "da": da, "a": a, "grandezza": "PCG", "limit": 999})
    mm = {m["data_misura"][:10]: m["valore"] for m in r["dati"] if m["valore"] is not None}
    return s, [mm.get(g) for g in giorni]  # None = dato mancante, diverso da 0 mm

to_ll = Transformer.from_crs(32632, 4326, always_xy=True)
stazioni = [s for s in get("stazioni", {}) if s["data_fine"] is None and s["wgs84_est"]]
with ThreadPoolExecutor(3) as ex:
    res = [(s, v) for s, v in ex.map(station, stazioni) if any(x is not None for x in v)]
out = {"giorni": giorni, "stazioni": [
    {"cod": s["cod_stazione"], "nome": s["nome"], "lon": round(lon, 5), "lat": round(lat, 5), "mm": v}
    for s, v in res for lon, lat in [to_ll.transform(s["wgs84_est"], s["wgs84_nord"])]]}
json.dump(out, open("data/rain.json", "w"), ensure_ascii=False)
print(f"{len(out['stazioni'])} stazioni con pioggia su {len(stazioni)} attive, {len(giorni)} giorni")
