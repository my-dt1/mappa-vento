"""Quota del terreno (m) sulla stessa griglia di contour_field.py, da tile Terrarium (AWS Open Data "Terrain Tiles",
dati SRTM e altre fonti aperte). Uso: python scripts/fetch_dem.py  ->  data/dem.npy (int16, righe da sud a nord)."""
import io, math, urllib.request
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

LON0, LON1, LAT0, LAT1, D = 8.0, 9.9, 38.8, 41.4, 0.005  # come in contour_field.py
Z = 9
N = 2 ** Z
gx, gy = np.meshgrid(np.arange(LON0, LON1 + D / 2, D), np.arange(LAT0, LAT1 + D / 2, D))
px = lambda lon: (lon + 180) / 360 * N * 256
py = lambda lat: (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * N * 256
x0, x1 = int(px(LON0) // 256), int(px(LON1) // 256)
y0, y1 = int(py(LAT1) // 256), int(py(LAT0) // 256)  # y cresce verso sud

mosaico = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256))
for tx in range(x0, x1 + 1):
    for ty in range(y0, y1 + 1):
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{tx}/{ty}.png"
        rgb = np.asarray(Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=60).read())).convert("RGB"), dtype=float)
        mosaico[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
quota = map_coordinates(mosaico, [py(gy) - y0 * 256 - .5, px(gx) - x0 * 256 - .5], order=1)  # -.5: i pixel sono centrati
quota = np.clip(quota, 0, None)  # il mare (batimetria) conta come quota 0
np.save("data/dem.npy", np.round(quota).astype(np.int16))
print("griglia", quota.shape, "quota min/max", quota.min().round(), quota.max().round())
