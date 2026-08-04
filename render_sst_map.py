"""
render_sst_map.py
Fetches NOAA OISST v2.1 highres daily SST anomaly via OPeNDAP.
Computes 7-day mean for Indian Ocean region.
Renders heatmap PNG with IOD pole boxes overlaid.
Outputs: output/sst_anomaly.png
         output/sst_poles.json

Run: python render_sst_map.py
Requires: pip install netCDF4 numpy matplotlib cartopy
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
from datetime import datetime, date

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "output")
PNG_PATH     = os.path.join(OUTPUT_DIR, "sst_anomaly.png")
POLES_PATH   = os.path.join(OUTPUT_DIR, "sst_poles.json")
THREDDS_BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"

# ── Grid helpers (0.25° grid) ────────────────────────────────
def lat_idx(lat): return round((lat + 89.875) / 0.25)
def lon_idx(lon): return round((lon -  0.125) / 0.25)

# Indian Ocean region: 30S–30N, 30E–130E
REG_LAT_S = lat_idx(-30);  REG_LAT_N = lat_idx(30)
REG_LON_W = lon_idx(30);   REG_LON_E = lon_idx(130)

# Pole box indices relative to region subset
def rel_lat(lat): return lat_idx(lat) - REG_LAT_S
def rel_lon(lon): return lon_idx(lon) - REG_LON_W

W_LAT = slice(rel_lat(-10), rel_lat(10)  + 1)
W_LON = slice(rel_lon(50),  rel_lon(70)  + 1)
E_LAT = slice(rel_lat(-10), rel_lat(0)   + 1)
E_LON = slice(rel_lon(90),  rel_lon(110) + 1)


def fetch_oisst():
    year = date.today().year
    base_url = f"{THREDDS_BASE}/sst.day.anom.{year}.nc"
    print(f"Connecting to OISST: sst.day.anom.{year}.nc ...")

    # Step 1: get time dimension size
    probe = nc.Dataset(
        f"{base_url}?anom[0:1:0][{REG_LAT_S}:1:{REG_LAT_S}][{REG_LON_W}:1:{REG_LON_W}]"
        f",time[0:1:0]"
    )
    n_time     = probe.dimensions['time'].size
    time_units = probe.variables['time'].units
    probe.close()
    print(f"  Total time steps: {n_time}")

    # Step 2: fetch last 7 days
    t_end   = n_time - 1
    t_start = max(0, n_time - 7)

    url = (
        f"{base_url}"
        f"?anom[{t_start}:1:{t_end}]"
        f"[{REG_LAT_S}:1:{REG_LAT_N}]"
        f"[{REG_LON_W}:1:{REG_LON_E}]"
        f",time[{t_start}:1:{t_end}]"
    )
    print(f"  Fetching last {n_time - t_start} days...")
    ds    = nc.Dataset(url)
    anom  = ds.variables['anom'][:]
    times = nc.num2date(ds.variables['time'][:], time_units)
    ds.close()
    print(f"  Shape: {anom.shape}  ({times[0]} → {times[-1]})")
    return anom, times


def compute_poles(mean_2d):
    west = float(np.ma.mean(mean_2d[W_LAT, W_LON]))
    east = float(np.ma.mean(mean_2d[E_LAT, E_LON]))
    return round(west, 3), round(east, 3), round(west - east, 3)


def render_map(anom_mean, times, west, east, dmi):
    """Render map — tries Cartopy first, falls back to plain Matplotlib."""

    lats = np.linspace(-30, 30,  anom_mean.shape[0])
    lons = np.linspace(30,  130, anom_mean.shape[1])
    vmax = 2.0
    cmap = plt.cm.RdBu_r
    date_str  = str(times[-1])[:10]
    start_str = str(times[0])[:10]

    rendered = False

    # ── Try Cartopy ───────────────────────────────────────────
    try:
        import cartopy.crs      as ccrs
        import cartopy.feature  as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

        fig = plt.figure(figsize=(13, 6), facecolor='#0a0f1a')
        ax  = fig.add_subplot(
            1, 1, 1,
            projection=ccrs.PlateCarree(central_longitude=80),
            facecolor='#0d1b2e'
        )

        # Set extent BEFORE adding features — avoids geometry errors
        ax.set_extent([30, 130, -30, 30], crs=ccrs.PlateCarree())

        # SST anomaly
        lon2d, lat2d = np.meshgrid(lons, lats)
        im = ax.pcolormesh(
            lon2d, lat2d, anom_mean,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=-vmax, vmax=vmax,
            shading='auto', zorder=1
        )

        # Use lower-resolution features to avoid geometry errors
        ax.add_feature(
            cfeature.NaturalEarthFeature('physical', 'land', '110m',
                facecolor='#1e293b', edgecolor='none'), zorder=2)
        ax.add_feature(
            cfeature.NaturalEarthFeature('physical', 'coastline', '110m',
                facecolor='none', edgecolor='#4a5568', linewidth=0.7), zorder=3)

        # IOD pole boxes
        # West pole rectangle
        west_rect = mpatches.Rectangle(
            (50, -10), 20, 20,
            fill=False, edgecolor='#f59e0b', linewidth=2.0,
            transform=ccrs.PlateCarree(), zorder=5
        )
        ax.add_patch(west_rect)

        # East pole rectangle
        east_rect = mpatches.Rectangle(
            (90, -10), 20, 10,
            fill=False, edgecolor='#818cf8', linewidth=2.0,
            transform=ccrs.PlateCarree(), zorder=5
        )
        ax.add_patch(east_rect)

        # Pole value labels
        ax.text(60, 13, f'W: {west:+.2f}°C',
                transform=ccrs.PlateCarree(), zorder=6,
                color='#f59e0b', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.75, ec='none'))
        ax.text(100, 2, f'E: {east:+.2f}°C',
                transform=ccrs.PlateCarree(), zorder=6,
                color='#818cf8', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.75, ec='none'))

        # Gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.3, color='#334155', alpha=0.8,
            xlocs=range(30, 131, 20), ylocs=range(-30, 31, 10)
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': '#64748b', 'fontsize': 8}
        gl.ylabel_style = {'color': '#64748b', 'fontsize': 8}

        rendered = True
        print("  Rendered with Cartopy")

    except Exception as e:
        print(f"  Cartopy render failed ({e}) — using fallback")
        plt.close('all')

    # ── Fallback: plain Matplotlib ────────────────────────────
    if not rendered:
        fig, ax = plt.subplots(figsize=(13, 6), facecolor='#0a0f1a')
        ax.set_facecolor('#0d1b2e')

        lon2d, lat2d = np.meshgrid(lons, lats)
        im = ax.pcolormesh(lon2d, lat2d, anom_mean,
                           cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')

        ax.set_xlim(30, 130)
        ax.set_ylim(-30, 30)
        ax.set_xlabel('Longitude', color='#64748b', fontsize=8)
        ax.set_ylabel('Latitude',  color='#64748b', fontsize=8)
        ax.tick_params(colors='#64748b', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

        # Pole boxes
        ax.add_patch(mpatches.Rectangle(
            (50, -10), 20, 20, fill=False, edgecolor='#f59e0b', linewidth=2))
        ax.add_patch(mpatches.Rectangle(
            (90, -10), 20, 10, fill=False, edgecolor='#818cf8', linewidth=2))

        # Labels
        ax.text(60, 12, f'W: {west:+.2f}°C',
                color='#f59e0b', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.75))
        ax.text(100, 1, f'E: {east:+.2f}°C',
                color='#818cf8', fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#0a0f1a', alpha=0.75))

        # Reference lines
        ax.axhline(0,  color='#334155', linewidth=0.5, linestyle='--')
        ax.axvline(80, color='#334155', linewidth=0.5, linestyle='--')

        print("  Rendered with Matplotlib fallback")

    # ── Shared: colorbar + title ──────────────────────────────
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        pad=0.02, fraction=0.025, extend='both')
    cbar.set_label('SST Anomaly (°C)', color='#94a3b8', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8')
    cbar.outline.set_edgecolor('#334155')

    phase_str = ("Positive IOD" if dmi >= 0.4
                 else "Negative IOD" if dmi <= -0.4
                 else "Neutral")
    fig.suptitle(
        f'Indian Ocean SST Anomaly  ·  DMI (W−E) = {dmi:+.3f}°C  [{phase_str}]\n'
        f'7-day mean  {start_str} → {date_str}  ·  Source: NOAA OISST v2.1 (0.25°)',
        color='#e2e8f0', fontsize=10, y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight',
                facecolor='#0a0f1a', edgecolor='none')
    plt.close()
    kb = os.path.getsize(PNG_PATH) / 1024
    print(f"  Saved: {PNG_PATH}  ({kb:.0f} KB)")


def main():
    print("=" * 50)
    print("IOD Pipeline — render_sst_map.py")
    print(f"Run time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    try:
        anom, times = fetch_oisst()

        anom_mean = np.ma.mean(anom, axis=0)
        print(f"  7-day mean shape: {anom_mean.shape}")

        west, east, dmi = compute_poles(anom_mean)
        date_str = str(times[-1])[:10]

        print(f"  West pole: {west:+.3f}°C")
        print(f"  East pole: {east:+.3f}°C")
        print(f"  Derived DMI: {dmi:+.3f}°C")

        render_map(anom_mean, times, west, east, dmi)

        poles = {
            "generated_utc": datetime.utcnow().isoformat() + "Z",
            "date":    date_str,
            "source":  "NOAA OISST v2.1 highres (0.25°)",
            "cadence": "7-day mean, updated weekly",
            "west_pole": {"region": "50–70°E, 10°S–10°N", "anomaly_c": west},
            "east_pole": {"region": "90–110°E, 10°S–0°N", "anomaly_c": east},
            "derived_dmi": dmi,
            "errors": errors
        }
        with open(POLES_PATH, "w") as f:
            json.dump(poles, f, indent=2)
        print(f"  Poles JSON: {POLES_PATH}")

    except Exception as e:
        msg = f"OISST render failed: {e}"
        print(f"  ERROR: {msg}")
        import traceback; traceback.print_exc()
        errors.append(msg)
        with open(POLES_PATH, "w") as f:
            json.dump({"generated_utc": datetime.utcnow().isoformat() + "Z",
                       "errors": errors}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
