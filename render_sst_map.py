"""
render_sst_map.py
Downloads last 30 daily OISST v2.1 files from NCEI.
Computes 7-day and 30-day SST anomaly means.
Renders two Indian Ocean heatmaps with IOD pole boxes.

Outputs:
  output/sst_anomaly_weekly.png   — 7-day mean
  output/sst_anomaly_monthly.png  — 30-day mean
  output/sst_poles.json           — pole values for both periods

Run: python render_sst_map.py
Requires: pip install requests numpy netCDF4 matplotlib cartopy
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
NCEI_BASE   = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum"
               "-interpolation/v2.1/access/avhrr")
HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; IOD-monitor/1.0)"}

# Indian Ocean region
LAT_S_DEG, LAT_N_DEG = -30,  30
LON_W_DEG, LON_E_DEG =  30, 130


def build_url(d):
    """Try preliminary filename first (recent), then final (older >2 weeks)."""
    ym  = d.strftime("%Y%m")
    ymd = d.strftime("%Y%m%d")
    today = date.today()
    if (today - d).days <= 14:
        return (f"{NCEI_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}_preliminary.nc",
                f"{NCEI_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}.nc")
    else:
        return (f"{NCEI_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}.nc",
                f"{NCEI_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}_preliminary.nc")


def download_file(d, tmpdir):
    """Download one daily OISST file. Returns local path or None."""
    urls = build_url(d)
    out  = os.path.join(tmpdir, f"oisst_{d.isoformat()}.nc")
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    f.write(chunk)
            size_mb = os.path.getsize(out) / 1024 / 1024
            print(f"  {d}  OK  ({size_mb:.1f} MB)  {url.split('/')[-1]}")
            return out
        except Exception:
            continue
    print(f"  {d}  SKIP")
    return None


def extract_anom(nc_path):
    """
    Extract Indian Ocean SST anomaly from one daily file.
    Uses set_auto_maskandscale(False) to handle raw int16 correctly.
    Returns 2D masked array (lat, lon) in °C.
    """
    ds       = nc.Dataset(nc_path)
    anom_var = ds.variables['anom']
    lats     = ds.variables['lat'][:]
    lons     = ds.variables['lon'][:]

    # Find Indian Ocean indices using actual lat/lon values
    lat_s = int(np.argmin(np.abs(lats - LAT_S_DEG)))
    lat_n = int(np.argmin(np.abs(lats - LAT_N_DEG)))
    lon_w = int(np.argmin(np.abs(lons - LON_W_DEG)))
    lon_e = int(np.argmin(np.abs(lons - LON_E_DEG)))

    # Disable auto scaling — handle manually to avoid fill value issues
    anom_var.set_auto_maskandscale(False)
    raw   = anom_var[0, 0, lat_s:lat_n+1, lon_w:lon_e+1]
    scale = float(anom_var.scale_factor)
    fill  = int(anom_var._FillValue)
    ds.close()

    masked = np.ma.masked_where(raw == fill, raw)
    return masked.astype(np.float32) * scale


def compute_poles(anom_2d, lats_subset, lons_subset):
    """Compute west and east pole box averages."""
    w_lat = np.where((lats_subset >= -10) & (lats_subset <= 10))[0]
    w_lon = np.where((lons_subset >=  50) & (lons_subset <= 70))[0]
    e_lat = np.where((lats_subset >= -10) & (lats_subset <=  0))[0]
    e_lon = np.where((lons_subset >=  90) & (lons_subset <= 110))[0]

    west_box = anom_2d[np.ix_(w_lat, w_lon)]
    east_box = anom_2d[np.ix_(e_lat, e_lon)]

    west = round(float(np.ma.mean(west_box)), 3)
    east = round(float(np.ma.mean(east_box)), 3)
    dmi  = round(west - east, 3)
    return west, east, dmi


def render_map(anom_2d, lats, lons, date_start, date_end,
               west, east, dmi, label, out_path):
    """Render one SST anomaly map and save to out_path."""
    lon2d, lat2d = np.meshgrid(lons, lats)
    vmax      = 2.0
    cmap      = plt.cm.RdBu_r
    n_days    = (date_end - date_start).days + 1
    phase_str = ("Positive IOD" if dmi >= 0.4
                 else "Negative IOD" if dmi <= -0.4 else "Neutral")
    rendered  = False

    try:
        import cartopy.crs     as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(13, 6), facecolor='#0a0f1a')
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(),
                              facecolor='#0d1b2e')
        ax.set_extent([LON_W_DEG, LON_E_DEG, LAT_S_DEG, LAT_N_DEG],
                      crs=ccrs.PlateCarree())

        im = ax.pcolormesh(lon2d, lat2d, anom_2d,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='auto', zorder=1)

        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'land', '110m',
            facecolor='#1e293b', edgecolor='none'), zorder=2)
        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'coastline', '110m',
            facecolor='none', edgecolor='#4a5568', linewidth=0.7), zorder=3)

        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     linewidth=0.3, color='#334155', alpha=0.7,
                     xlocs=range(30, 131, 20), ylocs=range(-30, 31, 10))

        ax.set_xticks(range(30, 131, 20), crs=ccrs.PlateCarree())
        ax.set_yticks(range(-30, 31, 10), crs=ccrs.PlateCarree())
        ax.set_xticklabels([f'{x}°E' for x in range(30, 131, 20)],
                           color='#64748b', fontsize=7.5)
        ax.set_yticklabels(
            [f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}'
             for y in range(-30, 31, 10)],
            color='#64748b', fontsize=7.5)
        ax.tick_params(color='#334155')

        # Pole boxes
        ax.add_patch(mpatches.Rectangle(
            (50, -10), 20, 20, fill=False, edgecolor='#f59e0b',
            linewidth=2.0, transform=ccrs.PlateCarree(), zorder=5))
        ax.add_patch(mpatches.Rectangle(
            (90, -10), 20, 10, fill=False, edgecolor='#818cf8',
            linewidth=2.0, transform=ccrs.PlateCarree(), zorder=5))

        ax.text(60, 14, f'W: {west:+.2f}°C',
                transform=ccrs.PlateCarree(), zorder=6,
                color='#f59e0b', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a',
                          alpha=0.8, ec='none'))
        ax.text(100, 2, f'E: {east:+.2f}°C',
                transform=ccrs.PlateCarree(), zorder=6,
                color='#818cf8', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a',
                          alpha=0.8, ec='none'))

        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

        rendered = True
        print(f"  [{label}] Rendered with Cartopy")

    except Exception as e:
        print(f"  [{label}] Cartopy failed ({e}) — Matplotlib fallback")
        plt.close('all')

    if not rendered:
        fig, ax = plt.subplots(figsize=(13, 6), facecolor='#0a0f1a')
        ax.set_facecolor('#0d1b2e')
        im = ax.pcolormesh(lon2d, lat2d, anom_2d,
                           cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlim(LON_W_DEG, LON_E_DEG)
        ax.set_ylim(LAT_S_DEG, LAT_N_DEG)
        ax.set_xticks(range(30, 131, 20))
        ax.set_yticks(range(-30, 31, 10))
        ax.set_xticklabels([f'{x}°E' for x in range(30, 131, 20)],
                           color='#64748b', fontsize=8)
        ax.set_yticklabels(
            [f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}'
             for y in range(-30, 31, 10)], color='#64748b', fontsize=8)
        ax.axhline(0, color='#334155', linewidth=0.4, linestyle='--')
        ax.add_patch(mpatches.Rectangle(
            (50,-10), 20, 20, fill=False, edgecolor='#f59e0b', linewidth=2))
        ax.add_patch(mpatches.Rectangle(
            (90,-10), 20, 10, fill=False, edgecolor='#818cf8', linewidth=2))
        ax.text(60, 14, f'W: {west:+.2f}°C', color='#f59e0b',
                fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.8))
        ax.text(100, 2, f'E: {east:+.2f}°C', color='#818cf8',
                fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.8))
        print(f"  [{label}] Rendered with Matplotlib fallback")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        pad=0.02, fraction=0.025, extend='both')
    cbar.set_label('SST Anomaly (°C)', color='#94a3b8', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8')
    cbar.outline.set_edgecolor('#334155')

    fig.suptitle(
        f'Indian Ocean SST Anomaly — {label}  ·  '
        f'DMI (W−E) = {dmi:+.3f}°C  [{phase_str}]\n'
        f'{n_days}-day mean  {date_start} → {date_end}  '
        f'·  NOAA OISST v2.1 (0.25°)  ·  Anom vs 1971-2000 clim',
        color='#e2e8f0', fontsize=10, y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0f1a', edgecolor='none')
    plt.close()
    kb = os.path.getsize(out_path) / 1024
    print(f"  [{label}] Saved: {out_path}  ({kb:.0f} KB)")


def main():
    print("=" * 55)
    print("IOD Pipeline — render_sst_map.py")
    print(f"Run time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    try:
        today      = date.today()
        # Start from 2 days ago (yesterday often not ready)
        # Collect up to 35 attempts to get 30 good files
        candidates = [today - timedelta(days=i) for i in range(2, 37)]

        print(f"Downloading up to 30 daily files "
              f"({candidates[0]} → {candidates[-1]})...")

        daily_anoms = []   # list of 2D arrays, newest first
        good_dates  = []
        lats_sub    = None
        lons_sub    = None

        with tempfile.TemporaryDirectory() as tmpdir:
            for d in candidates:
                if len(good_dates) >= 30:
                    break
                path = download_file(d, tmpdir)
                if not path:
                    continue

                try:
                    # Get lat/lon from first file
                    if lats_sub is None:
                        ds = nc.Dataset(path)
                        lats_all = ds.variables['lat'][:]
                        lons_all = ds.variables['lon'][:]
                        lat_s = int(np.argmin(np.abs(lats_all - LAT_S_DEG)))
                        lat_n = int(np.argmin(np.abs(lats_all - LAT_N_DEG)))
                        lon_w = int(np.argmin(np.abs(lons_all - LON_W_DEG)))
                        lon_e = int(np.argmin(np.abs(lons_all - LON_E_DEG)))
                        lats_sub = np.array(lats_all[lat_s:lat_n+1])
                        lons_sub = np.array(lons_all[lon_w:lon_e+1])
                        ds.close()

                    anom = extract_anom(path)
                    daily_anoms.append(anom)
                    good_dates.append(d)

                except Exception as e:
                    print(f"  {d}  extract failed: {e}")
                    continue

        if len(good_dates) < 7:
            raise RuntimeError(
                f"Only {len(good_dates)} files — need at least 7")

        good_dates = sorted(good_dates)   # chronological
        daily_anoms = list(reversed(daily_anoms))  # now oldest→newest

        print(f"\nGot {len(good_dates)} files: "
              f"{good_dates[0]} → {good_dates[-1]}")

        # 7-day mean (last 7)
        anom_7d     = np.ma.mean(np.ma.stack(daily_anoms[-7:]),  axis=0)
        dates_7d    = good_dates[-7:]

        # 30-day mean (all)
        anom_30d    = np.ma.mean(np.ma.stack(daily_anoms),        axis=0)
        dates_30d   = good_dates

        print(f"\n7-day  mean shape: {anom_7d.shape}")
        print(f"30-day mean shape: {anom_30d.shape}")

        # Pole values
        w7,  e7,  d7  = compute_poles(anom_7d,  lats_sub, lons_sub)
        w30, e30, d30 = compute_poles(anom_30d, lats_sub, lons_sub)

        print(f"\nWeekly  — W:{w7:+.3f}  E:{e7:+.3f}  DMI:{d7:+.3f}°C")
        print(f"Monthly — W:{w30:+.3f}  E:{e30:+.3f}  DMI:{d30:+.3f}°C")

        # Render weekly map
        render_map(
            anom_7d, lats_sub, lons_sub,
            dates_7d[0], dates_7d[-1],
            w7, e7, d7,
            label="7-Day Mean",
            out_path=os.path.join(OUTPUT_DIR, "sst_anomaly_weekly.png")
        )

        # Render monthly map
        render_map(
            anom_30d, lats_sub, lons_sub,
            dates_30d[0], dates_30d[-1],
            w30, e30, d30,
            label="30-Day Mean",
            out_path=os.path.join(OUTPUT_DIR, "sst_anomaly_monthly.png")
        )

        # Write poles JSON
        poles = {
            "generated_utc": datetime.utcnow().isoformat() + "Z",
            "source":        "NOAA OISST v2.1 (NCEI daily files, 0.25°)",
            "climatology":   "1971-2000 (embedded in OISST anom variable)",
            "weekly": {
                "date_start":  dates_7d[0].isoformat(),
                "date_end":    dates_7d[-1].isoformat(),
                "n_days":      len(dates_7d),
                "west_pole":   {"region": "50–70°E, 10°S–10°N", "anomaly_c": w7},
                "east_pole":   {"region": "90–110°E, 10°S–0°N", "anomaly_c": e7},
                "derived_dmi": d7
            },
            "monthly": {
                "date_start":  dates_30d[0].isoformat(),
                "date_end":    dates_30d[-1].isoformat(),
                "n_days":      len(dates_30d),
                "west_pole":   {"region": "50–70°E, 10°S–10°N", "anomaly_c": w30},
                "east_pole":   {"region": "90–110°E, 10°S–0°N", "anomaly_c": e30},
                "derived_dmi": d30
            },
            "errors": errors
        }
        poles_path = os.path.join(OUTPUT_DIR, "sst_poles.json")
        with open(poles_path, "w") as f:
            json.dump(poles, f, indent=2)
        print(f"\nPoles JSON: {poles_path}")

    except Exception as e:
        msg = f"SST render failed: {e}"
        print(f"\nERROR: {msg}")
        import traceback; traceback.print_exc()
        errors.append(msg)
        with open(os.path.join(OUTPUT_DIR, "sst_poles.json"), "w") as f:
            json.dump({
                "generated_utc": datetime.utcnow().isoformat() + "Z",
                "errors": errors
            }, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
