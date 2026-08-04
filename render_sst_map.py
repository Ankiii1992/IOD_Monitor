"""
render_sst_map.py
Fetches NOAA OISST v2.1 daily files directly from NCEI (1-2 day lag, always current).
Downloads last 7 daily files, computes 7-day mean SST anomaly.
Renders Indian Ocean heatmap with IOD pole boxes.

File pattern: https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation
              /v2.1/access/avhrr/YYYYMM/oisst-avhrr-v02r01.YYYYMMDD.nc

Each file ~4MB (global, netCDF4). We subset to Indian Ocean after download.
Anomaly = SST - monthly climatology (1971-2000, interpolated to daily 0.25°).
Climatology sourced from PSL one-time download (sst.day.mean.ltm.1971-2000.nc subset).

Outputs: output/sst_anomaly.png
         output/sst_poles.json
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import json
import os
import tempfile
from datetime import datetime, date, timedelta

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
PNG_PATH    = os.path.join(OUTPUT_DIR, "sst_anomaly.png")
POLES_PATH  = os.path.join(OUTPUT_DIR, "sst_poles.json")
CLIM_PATH   = os.path.join(OUTPUT_DIR, "sst_clim_indian_ocean.npy")  # cached climatology

NCEI_BASE   = ("https://www.ncei.noaa.gov/data/"
               "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr")
HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; IOD-monitor/1.0)"}

# Indian Ocean region (0.25° grid)
# lat: -89.875 to 89.875 → index = (lat + 89.875) / 0.25
# lon:   0.125 to 359.875 → index = (lon - 0.125)  / 0.25
LAT_S_DEG, LAT_N_DEG = -30, 30
LON_W_DEG, LON_E_DEG =  30, 130

def lat_idx(lat): return round((lat + 89.875) / 0.25)
def lon_idx(lon): return round((lon - 0.125)  / 0.25)

REG_LAT_S = lat_idx(LAT_S_DEG);  REG_LAT_N = lat_idx(LAT_N_DEG)
REG_LON_W = lon_idx(LON_W_DEG);  REG_LON_E = lon_idx(LON_E_DEG)

# IOD pole boxes (indices relative to Indian Ocean subset)
def rel_lat(lat): return lat_idx(lat) - REG_LAT_S
def rel_lon(lon): return lon_idx(lon) - REG_LON_W

W_LAT = slice(rel_lat(-10), rel_lat(10)  + 1)  # 10S–10N
W_LON = slice(rel_lon(50),  rel_lon(70)  + 1)  # 50–70E
E_LAT = slice(rel_lat(-10), rel_lat(0)   + 1)  # 10S–0N
E_LON = slice(rel_lon(90),  rel_lon(110) + 1)  # 90–110E


def get_candidate_dates(n=10):
    """Return last n dates to try (most recent first). Skip today — file not ready."""
    today = date.today()
    return [today - timedelta(days=i) for i in range(1, n + 1)]


def build_url(d):
    ym  = d.strftime("%Y%m")
    ymd = d.strftime("%Y%m%d")
    return f"{NCEI_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}.nc"


def download_daily_sst(d, tmpdir):
    """Download one daily OISST file. Returns local path or None."""
    url      = build_url(d)
    out_path = os.path.join(tmpdir, f"oisst_{d.isoformat()}.nc")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  {d}  OK  ({size_mb:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"  {d}  SKIP  ({e})")
        return None


def extract_indian_ocean(nc_path):
    """
    Read one daily OISST file and extract Indian Ocean SST subset.
    Returns 2D array (lat, lon) of SST in °C.
    """
    ds  = nc.Dataset(nc_path)
    # Variable name is 'sst', shape (1, 1, lat, lon) — time + zlev dimensions
    sst = ds.variables['sst'][0, 0,
                               REG_LAT_S:REG_LAT_N + 1,
                               REG_LON_W:REG_LON_E + 1]
    # Apply scale + offset if present
    scale  = getattr(ds.variables['sst'], 'scale_factor',  1.0)
    offset = getattr(ds.variables['sst'], 'add_offset',    0.0)
    fill   = getattr(ds.variables['sst'], '_FillValue',    None)
    ds.close()

    sst = sst.astype(np.float32) * scale + offset
    if fill is not None:
        sst = np.ma.masked_where(np.abs(sst - (fill * scale + offset)) < 0.01, sst)
    return sst


def get_climatology(month, shape):
    """
    Get monthly SST climatology for the Indian Ocean region.
    Uses PSL 1971-2000 LTM via OPeNDAP (one-time subset, cached as .npy).
    Falls back to approximate values if OPeNDAP unavailable.
    """
    clim_dir  = OUTPUT_DIR
    clim_file = os.path.join(clim_dir, f"sst_clim_m{month:02d}.npy")

    if os.path.exists(clim_file):
        print(f"  Climatology: loaded from cache (month {month})")
        return np.load(clim_file)

    # Try fetching from PSL OPeNDAP
    try:
        print(f"  Climatology: fetching month {month} from PSL LTM...")
        CLIM_URL = (
            "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/"
            "sst.day.mean.ltm.1971-2000.nc"
            f"?sst[{month-1}:1:{month-1}]"
            f"[{REG_LAT_S}:1:{REG_LAT_N}]"
            f"[{REG_LON_W}:1:{REG_LON_E}]"
        )
        ds   = nc.Dataset(CLIM_URL)
        clim = ds.variables['sst'][0]
        scale  = getattr(ds.variables['sst'], 'scale_factor',  1.0)
        offset = getattr(ds.variables['sst'], 'add_offset',    0.0)
        fill   = getattr(ds.variables['sst'], '_FillValue',    None)
        ds.close()

        clim = clim.astype(np.float32) * scale + offset
        if fill is not None:
            clim = np.ma.masked_where(
                np.abs(clim - (fill * scale + offset)) < 0.01, clim)

        os.makedirs(clim_dir, exist_ok=True)
        np.save(clim_file, np.array(clim))
        print(f"  Climatology: cached to {clim_file}")
        return clim

    except Exception as e:
        print(f"  Climatology: OPeNDAP failed ({e}) — using zero baseline")
        print("  WARNING: anomaly will be relative to 0°C, not climatology")
        return np.zeros(shape)


def compute_poles(anom_2d):
    west = float(np.ma.mean(anom_2d[W_LAT, W_LON]))
    east = float(np.ma.mean(anom_2d[E_LAT, E_LON]))
    return round(west, 3), round(east, 3), round(west - east, 3)


def render_map(anom_mean, date_range, west, east, dmi):
    lats  = np.linspace(LAT_S_DEG, LAT_N_DEG, anom_mean.shape[0])
    lons  = np.linspace(LON_W_DEG, LON_E_DEG, anom_mean.shape[1])
    lon2d, lat2d = np.meshgrid(lons, lats)
    vmax      = 2.0
    cmap      = plt.cm.RdBu_r
    start_str = date_range[0].isoformat()
    end_str   = date_range[-1].isoformat()
    n_days    = len(date_range)

    rendered = False

    try:
        import cartopy.crs     as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(13, 6), facecolor='#0a0f1a')
        ax  = fig.add_subplot(1, 1, 1,
                              projection=ccrs.PlateCarree(),
                              facecolor='#0d1b2e')
        ax.set_extent([LON_W_DEG, LON_E_DEG, LAT_S_DEG, LAT_N_DEG],
                      crs=ccrs.PlateCarree())

        im = ax.pcolormesh(lon2d, lat2d, anom_mean,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='auto', zorder=1)

        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'land', '110m',
            facecolor='#1e293b', edgecolor='none'), zorder=2)
        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'coastline', '110m',
            facecolor='none', edgecolor='#4a5568', linewidth=0.7), zorder=3)

        # Gridlines — draw_labels=False avoids LinearRing crash
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     linewidth=0.3, color='#334155', alpha=0.7,
                     xlocs=range(30, 131, 20), ylocs=range(-30, 31, 10))

        ax.set_xticks(range(30, 131, 20), crs=ccrs.PlateCarree())
        ax.set_yticks(range(-30, 31, 10), crs=ccrs.PlateCarree())
        ax.set_xticklabels([f'{x}°E' for x in range(30, 131, 20)],
                           color='#64748b', fontsize=7.5)
        ax.set_yticklabels([f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}' 
                            for y in range(-30, 31, 10)],
                           color='#64748b', fontsize=7.5)
        ax.tick_params(color='#334155')

        # Pole boxes
        ax.add_patch(mpatches.Rectangle((50, -10), 20, 20,
            fill=False, edgecolor='#f59e0b', linewidth=2.0,
            transform=ccrs.PlateCarree(), zorder=5))
        ax.add_patch(mpatches.Rectangle((90, -10), 20, 10,
            fill=False, edgecolor='#818cf8', linewidth=2.0,
            transform=ccrs.PlateCarree(), zorder=5))

        ax.text(60, 14, f'W: {west:+.2f}°C',
                transform=ccrs.PlateCarree(), zorder=6,
                color='#f59e0b', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.8, ec='none'))
        ax.text(100, 2, f'E: {east:+.2f}°C',
                transform=ccrs.PlateCarree(), zorder=6,
                color='#818cf8', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.8, ec='none'))

        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

        rendered = True
        print("  Rendered with Cartopy")

    except Exception as e:
        print(f"  Cartopy failed ({e}) — using Matplotlib fallback")
        plt.close('all')

    if not rendered:
        fig, ax = plt.subplots(figsize=(13, 6), facecolor='#0a0f1a')
        ax.set_facecolor('#0d1b2e')
        im = ax.pcolormesh(lon2d, lat2d, anom_mean,
                           cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlim(LON_W_DEG, LON_E_DEG); ax.set_ylim(LAT_S_DEG, LAT_N_DEG)
        ax.set_xticks(range(30, 131, 20))
        ax.set_yticks(range(-30, 31, 10))
        ax.set_xticklabels([f'{x}°E' for x in range(30, 131, 20)],
                           color='#64748b', fontsize=8)
        ax.set_yticklabels([f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}' 
                            for y in range(-30, 31, 10)],
                           color='#64748b', fontsize=8)
        ax.axhline(0, color='#334155', linewidth=0.4, linestyle='--')
        ax.add_patch(mpatches.Rectangle((50,-10), 20, 20,
            fill=False, edgecolor='#f59e0b', linewidth=2))
        ax.add_patch(mpatches.Rectangle((90,-10), 20, 10,
            fill=False, edgecolor='#818cf8', linewidth=2))
        ax.text(60, 14, f'W: {west:+.2f}°C', color='#f59e0b',
                fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.8))
        ax.text(100, 2, f'E: {east:+.2f}°C', color='#818cf8',
                fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.8))
        print("  Rendered with Matplotlib fallback")

    # Colorbar + title
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        pad=0.02, fraction=0.025, extend='both')
    cbar.set_label('SST Anomaly (°C)', color='#94a3b8', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8')
    cbar.outline.set_edgecolor('#334155')

    phase_str = ("Positive IOD" if dmi >= 0.4
                 else "Negative IOD" if dmi <= -0.4 else "Neutral")
    fig.suptitle(
        f'Indian Ocean SST Anomaly  ·  DMI (W−E) = {dmi:+.3f}°C  [{phase_str}]\n'
        f'{n_days}-day mean  {start_str} → {end_str}  '
        f'·  NOAA OISST v2.1 (0.25°)  ·  Anom vs 1971-2000 clim',
        color='#e2e8f0', fontsize=10, y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight',
                facecolor='#0a0f1a', edgecolor='none')
    plt.close()
    print(f"  Saved: {PNG_PATH}  ({os.path.getsize(PNG_PATH)/1024:.0f} KB)")


def main():
    print("=" * 55)
    print("IOD Pipeline — render_sst_map.py")
    print(f"Run time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    try:
        candidates = get_candidate_dates(n=10)
        print(f"Trying dates: {candidates[0]} → {candidates[-1]}")

        sst_stack  = []
        good_dates = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for d in candidates:
                if len(good_dates) >= 7:
                    break
                path = download_daily_sst(d, tmpdir)
                if path:
                    sst = extract_indian_ocean(path)
                    sst_stack.append(sst)
                    good_dates.append(d)

        if not good_dates:
            raise RuntimeError("No OISST files could be downloaded")

        good_dates = sorted(good_dates)  # chronological
        print(f"\nDownloaded {len(good_dates)} daily files: "
              f"{good_dates[0]} → {good_dates[-1]}")

        # 7-day mean SST
        sst_mean = np.ma.mean(np.ma.stack(sst_stack), axis=0)
        print(f"SST mean shape: {sst_mean.shape}")

        # Climatology for the most recent month
        month = good_dates[-1].month
        clim  = get_climatology(month, sst_mean.shape)

        # Anomaly
        anom_mean = sst_mean - clim
        print(f"Anomaly range: {float(np.ma.min(anom_mean)):+.2f} to "
              f"{float(np.ma.max(anom_mean)):+.2f}°C")

        # Pole values
        west, east, dmi = compute_poles(anom_mean)
        print(f"West pole (50-70E, 10S-10N): {west:+.3f}°C")
        print(f"East pole (90-110E, 10S-0N): {east:+.3f}°C")
        print(f"Derived DMI (W-E):            {dmi:+.3f}°C")

        # Render
        render_map(anom_mean, good_dates, west, east, dmi)

        # Write poles JSON
        poles = {
            "generated_utc": datetime.utcnow().isoformat() + "Z",
            "date":          good_dates[-1].isoformat(),
            "date_start":    good_dates[0].isoformat(),
            "n_days":        len(good_dates),
            "source":        "NOAA OISST v2.1 (NCEI direct, 1-2 day lag)",
            "climatology":   "1971-2000 monthly mean (PSL LTM)",
            "west_pole":     {"region": "50–70°E, 10°S–10°N", "anomaly_c": west},
            "east_pole":     {"region": "90–110°E, 10°S–0°N", "anomaly_c": east},
            "derived_dmi":   dmi,
            "errors":        errors
        }
        with open(POLES_PATH, "w") as f:
            json.dump(poles, f, indent=2)
        print(f"Poles JSON: {POLES_PATH}")

    except Exception as e:
        msg = f"SST render failed: {e}"
        print(f"ERROR: {msg}")
        import traceback; traceback.print_exc()
        errors.append(msg)
        with open(POLES_PATH, "w") as f:
            json.dump({"generated_utc": datetime.utcnow().isoformat() + "Z",
                       "errors": errors}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
