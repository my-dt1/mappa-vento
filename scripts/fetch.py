"""Scarica direzione (DVI) e intensità (VAI) del vento per un giorno, da tutte le stazioni attive.
Uso: WIND_API_KEY=... python scripts/fetch.py 2026-03-01
Scrive data/raw/obs-<giorno>.json: lista di {cod, nome, est, nord, dati:[{t, dir, vel}]}."""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

B = "https://hmj6uzslsb.execute-api.eu-north-1.amazonaws.com/v1"
KEY = os.environ["WIND_API_KEY"]

def get(path, params):
    url = f"{B}/{path}?" + urllib.parse.urlencode(params, doseq=True)
    for i in range(6):  # l'API risponde 429/500 sotto carico
        try:
            req = urllib.request.Request(url, headers={"x-api-key": KEY})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            err = e; time.sleep(2 * (i + 1))
    raise err

def station(s, day):
    rows, cur = [], None
    while True:
        p = {"stazione": s["cod_stazione"], "da": day, "a": day, "limit": 999}
        r = get("misure", {"cursore": cur} if cur else p)
        rows += r["dati"]; cur = r["cursore_successivo"]
        if not cur: break
    by_t = {}
    for m in rows:
        if m["cod_grandezza"] not in ("DVI", "VAI"): continue  # l'API onora un solo filtro grandezza: filtro qui
        by_t.setdefault(m["data_misura"], {})[m["cod_grandezza"]] = m["valore"]
    dati = [{"t": t, "dir": v["DVI"], "vel": v["VAI"]} for t, v in sorted(by_t.items())
            if v.get("DVI") is not None and v.get("VAI") is not None]
    return {"cod": s["cod_stazione"], "nome": s["nome"], "est": s["wgs84_est"], "nord": s["wgs84_nord"], "dati": dati}

day = sys.argv[1]
stazioni = [s for s in get("stazioni", {}) if s["data_fine"] is None and s["wgs84_est"]]
with ThreadPoolExecutor(3) as ex:
    out = [o for o in ex.map(lambda s: station(s, day), stazioni) if o["dati"]]
json.dump(out, open(f"data/raw/obs-{day}.json", "w"), ensure_ascii=False)
print(f"{day}: {len(out)} stazioni con vento su {len(stazioni)} attive")
