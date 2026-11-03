"""Converte data/wind-<giorno>.json (griglie u/v orarie di interpolate.py) in COG a due bande per l'SDK JS di ESRI.
Uso: python scripts/make_uv_cog.py 2026-03-01  ->  esri/data/uv/<giorno>T<ora>.tif
Banda 1 = u (verso est, m/s), banda 2 = v (verso nord, m/s). Campi levigati, raddoppiati di risoluzione e
resi nodata a piu' di ~10 km dalla Sardegna (il mare lontano non ha stazioni: sarebbe vento inventato). Il tag DataType=Vector-UV fa riconoscere il raster
come flusso a FlowRenderer (da verificare su browser: fonte community, non documentazione ufficiale)."""
import json, os, sys
import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter, zoom
from shapely.geometry import shape
from rasterio.shutil import copy
from rasterio.transform import from_origin

giorno = sys.argv[1]
os.makedirs("esri/data/uv", exist_ok=True)
SIGMA = 1.5     # celle del grigliato originale (0.05 gradi): ~8 km, toglie i mulinelli dell'IDW su pochi punti
BUFFER = 0.1    # gradi (~10 km): oltre questa distanza dalla costa non si disegna
NODATA = -9999.0
terra = shapely.make_valid(shape(json.load(open("data/sardegna.geojson"))["geometry"])).buffer(BUFFER)
for u, v in json.load(open(f"data/wind-{giorno}.json")):
    h = u["header"]
    nx, ny = h["nx"], h["ny"]
    uv = np.stack([np.array(b["data"], dtype="float32").reshape(ny, nx) for b in (u, v)])  # righe da nord a sud
    # levigo e raddoppio la risoluzione (cubica): le linee di flusso sono piu' morbide
    uv = np.stack([zoom(gaussian_filter(b, SIGMA, mode="nearest"), 2, order=3, grid_mode=True, mode="nearest") for b in uv]).astype("float32")
    dx, dy = h["dx"] / 2, h["dy"] / 2
    # i punti griglia sono centri cella: l'origine del raster e' mezza cella a nord-ovest
    tr = from_origin(h["lo1"] - h["dx"] / 2, h["la1"] + h["dy"] / 2, dx, dy)
    vicino = rasterize([(terra, 1)], out_shape=uv.shape[1:], transform=tr, all_touched=True).astype(bool)
    uv[:, ~vicino] = NODATA
    prof = dict(driver="GTiff", width=uv.shape[2], height=uv.shape[1], count=2, dtype="float32", crs="EPSG:4326",
                transform=tr, nodata=NODATA)
    out = f"esri/data/uv/{h['refTime'][:13]}.tif"
    with rasterio.MemoryFile() as mem:
        with mem.open(**prof) as dst:
            dst.write(uv)
            dst.update_tags(DataType="Vector-UV")
            dst.set_band_description(1, "u"); dst.set_band_description(2, "v")
        with mem.open() as src:
            copy(src, out, driver="COG", compress="DEFLATE", predictor="3")
    with rasterio.open(out) as chk:  # verifica a posteriori: stesso contenuto e metadati
        assert chk.count == 2 and chk.tags().get("DataType") == "Vector-UV"
        assert np.allclose(chk.read(), uv)
print(giorno, "ok")
