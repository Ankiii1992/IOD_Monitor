"""
render_sst_map.py
Fetches NOAA OISST v2.1 highres daily SST anomaly via OPeNDAP.
Computes 7-day mean for Indian Ocean region.
Renders heatmap PNG with IOD pole boxes overlaid.
Outputs: output/sst_anomaly.png
         output/sst_poles.json  (west/east pole values for frontend)

Run: python render_sst_map.py
Requires: pip install netCDF4 numpy matplotlib cartopy
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import json
import os
from datetime import datetime, date

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "output")
PNG_PATH     = os.path.join(OUTPUT_DIR, "sst_anomaly.png")
POLES_PATH   = os.path.join(OUTPUT_DIR, "sst_poles.json")
THREDDS_BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"

# ── Grid helpers (0.25° grid) ────────────────────────────────
# lat: -89.875 to 89.875  → index = (lat + 89.875) / 0.25
# lon:   0.125 to 359.875 → index = (lon -  0.125) / 0.25
def lat_idx(lat): return round((lat + 89.875) / 0.25)
def lon_idx(lon): return round((lon -  0.125) / 0.25)

# Indian Ocean region to download
# lat: 30S to 30N, lon: 30E to 130E
REG_LAT_S = lat_idx(-30);  REG_LAT_N = lat_idx(30)
REG_LON_W = lon_idx(30);   REG_LON_E = lon_idx(130)

# IOD pole boxes (indices relative to region subset)
def rel_lat(lat): return lat_idx(lat) - REG_LAT_S
def rel_lon(lon): return lon_idx(lon) - REG_LON_W

# West pole: 50–70E, 10S–10N
W_LAT = slice(rel_lat(-10), rel_lat(10)  + 1)
W_LON = slice(rel_lon(50),  rel_lon(70)  + 1)
# East pole: 90–110E, 10S–0N
E_LAT = slice(rel_lat(-10), rel_lat(0)   + 1)
E_LON = slice(rel_lon(90),  rel_lon(110) + 1)


def fetch_oisst():
    """Fetch last 7 days of OISST anomaly for Indian Ocean region."""
    year = date.today().year
    base_url = f"{THREDDS_BASE}/sst.day.anom.{year}.nc"

    print(f"Connecting to OISST: sst.day.anom.{year}.nc ...")

    # Step 1: get total time dimension size
    probe = nc.Dataset(
        f"{base_url}?anom[0:1:0][{REG_LAT_S}:1:{REG_LAT_S}][{REG_LON_W}:1:{REG_LON_W}]"
        f",time[0:1:0]"
    )
    n_time     = probe.dimensions['time'].size
    time_units = probe.variables['time'].units
    probe.close()
    print(f"  File has {n_time} daily time steps in {year}")

    # Step 2: fetch last 7 days for Indian Ocean box
    t_end   = n_time - 1
    t_start = max(0, n_time - 7)

    url = (
        f"{base_url}"
        f"?anom[{t_start}:1:{t_end}]"
        f"[{REG_LAT_S}:1:{REG_LAT_N}]"
        f"[{REG_LON_W}:1:{REG_LON_E}]"
        f",time[{t_start}:1:{t_end}]"
    )

    print(f"  Fetching {n_time - t_start} days, Indian Ocean box...")
    ds    = nc.Dataset(url)
    anom  = ds.variables['anom'][:]          # (days, lat, lon)
    times = nc.num2date(ds.variables['time'][:], time_units)
    ds.close()

    print(f"  Shape: {anom.shape}  ({times[0]} → {times[-1]})")
    return anom, times


def compute_poles(anom_7day_mean):
    """Compute west and east pole box averages from 7-day mean."""
    west = float(np.ma.mean(anom_7day_mean[W_LAT, W_LON]))
    east = float(np.ma.mean(anom_7day_mean[E_LAT, E_LON]))
    dmi  = round(west - east, 3)
    return round(west, 3), round(east, 3), dmi


def render_map(anom_mean, times, west, east, dmi):
    """Render Indian Ocean SST anomaly heatmap with pole boxes."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False
        print("  Cartopy not available — rendering simple map")

    # Coordinate arrays for the subset
    lats = np.linspace(-30, 30, anom_mean.shape[0])
    lons = np.linspace(30, 130, anom_mean.shape[1])

    # Colour scale: symmetric around 0, capped at ±2°C
    vmax = 2.0
    cmap = plt.cm.RdBu_r

    date_str  = str(times[-1])[:10]
    start_str = str(times[0])[:10]

    if USE_CARTOPY:
        fig = plt.figure(figsize=(12, 6), facecolor='#0a0f1a')
        ax  = fig.add_subplot(1, 1, 1,
                              projection=ccrs.PlateCarree(),
                              facecolor='#0a0f1a')
        ax.set_extent([30, 130, -30, 30], crs=ccrs.PlateCarree())

        # SST anomaly filled contour
        im = ax.pcolormesh(lons, lats, anom_mean,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='auto')

        # Coastlines and land
        ax.add_feature(cfeature.LAND,       facecolor='#1a2236', zorder=2)
        ax.add_feature(cfeature.COASTLINE,  edgecolor='#4a5568', linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS,    edgecolor='#2d3748', linewidth=0.3, zorder=3)

        # IOD pole boxes
        # West pole: 50-70E, 10S-10N
        ax.add_patch(mpatches.Rectangle(
            (50, -10), 20, 20,
            fill=False, edgecolor='#f59e0b', linewidth=2,
            transform=ccrs.PlateCarree(), zorder=4,
            label=f'W pole: {west:+.2f}°C'
        ))
        # East pole: 90-110E, 10S-0N
        ax.add_patch(mpatches.Rectangle(
            (90, -10), 20, 10,
            fill=False, edgecolor='#6366f1', linewidth=2,
            transform=ccrs.PlateCarree(), zorder=4,
            label=f'E pole: {east:+.2f}°C'
        ))

        # Pole labels
        ax.text(60, 12, f'W: {west:+.2f}°C',
                color='#f59e0b', fontsize=9, fontweight='bold',
                transform=ccrs.PlateCarree(), zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0f1a', alpha=0.7))
        ax.text(95, 2, f'E: {east:+.2f}°C',
                color='#6366f1', fontsize=9, fontweight='bold',
                transform=ccrs.PlateCarree(), zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0f1a', alpha=0.7))

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color='#2d3748', alpha=0.8,
                          xlocs=range(30, 131, 20),
                          ylocs=range(-30, 31, 10))
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'color': '#94a3b8', 'fontsize': 8}
        gl.ylabel_style = {'color': '#94a3b8', 'fontsize': 8}

    else:
        # Simple fallback without Cartopy
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0f1a')
        ax.set_facecolor('#0a0f1a')
        im = ax.pcolormesh(lons, lats, anom_mean,
                           cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlim(30, 130); ax.set_ylim(-30, 30)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3748')

        # Pole boxes
        ax.add_patch(mpatches.Rectangle((50,-10),20,20,
            fill=False, edgecolor='#f59e0b', linewidth=2))
        ax.add_patch(mpatches.Rectangle((90,-10),20,10,
            fill=False, edgecolor='#6366f1', linewidth=2))
        ax.text(60, 11, f'W: {west:+.2f}°C',
                color='#f59e0b', fontsize=9, fontweight='bold')
        ax.text(91, 1,  f'E: {east:+.2f}°C',
                color='#6366f1', fontsize=9, fontweight='bold')

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        pad=0.02, fraction=0.02, extend='both')
    cbar.set_label('SST Anomaly (°C)', color='#94a3b8', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=8)
    cbar.outline.set_edgecolor('#2d3748')

    # Title
    phase_color = '#f59e0b' if dmi >= 0.4 else '#6366f1' if dmi <= -0.4 else '#94a3b8'
    fig.suptitle(
        f'Indian Ocean SST Anomaly  |  DMI (W−E) = {dmi:+.3f}°C\n'
        f'7-day mean: {start_str} → {date_str}  |  Source: NOAA OISST v2.1',
        color='#e2e8f0', fontsize=10, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight',
                facecolor='#0a0f1a', edgecolor='none')
    plt.close()
    print(f"  Map saved: {PNG_PATH}  "
          f"({os.path.getsize(PNG_PATH)/1024:.0f} KB)")


def main():
    print("=" * 50)
    print("IOD Pipeline — render_sst_map.py")
    print(f"Run time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    errors = []

    try:
        # Fetch OISST
        anom, times = fetch_oisst()

        # 7-day mean
        anom_mean = np.ma.mean(anom, axis=0)   # (lat, lon)
        print(f"  7-day mean computed: {anom_mean.shape}")

        # Pole values
        west, east, dmi = compute_poles(anom_mean)
        date_str = str(times[-1])[:10]
        print(f"  West pole (50-70E, 10S-10N): {west:+.3f}°C")
        print(f"  East pole (90-110E, 10S-0N): {east:+.3f}°C")
        print(f"  Derived DMI (W-E):            {dmi:+.3f}°C")

        # Render map
        render_map(anom_mean, times, west, east, dmi)

        # Write poles JSON
        poles = {
            "generated_utc": datetime.utcnow().isoformat() + "Z",
            "date":          date_str,
            "source":        "NOAA OISST v2.1 highres (0.25°)",
            "cadence":       "7-day mean, updated weekly",
            "west_pole": {
                "region":    "50–70°E, 10°S–10°N",
                "anomaly_c": west
            },
            "east_pole": {
                "region":    "90–110°E, 10°S–0°N",
                "anomaly_c": east
            },
            "derived_dmi":   dmi,
            "errors":        errors
        }
        with open(POLES_PATH, "w") as f:
            json.dump(poles, f, indent=2)
        print(f"  Poles JSON saved: {POLES_PATH}")

    except Exception as e:
        msg = f"OISST render failed: {e}"
        print(f"  ERROR: {msg}")
        import traceback; traceback.print_exc()
        errors.append(msg)

        # Write error JSON so frontend knows
        with open(POLES_PATH, "w") as f:
            json.dump({
                "generated_utc": datetime.utcnow().isoformat() + "Z",
                "errors": errors
            }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
