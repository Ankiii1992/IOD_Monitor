"""
render_sst_map.py
Downloads last 30 daily CoralTemp v3.1 SSTA files from NCEI.
Computes 7-day and 30-day SST anomaly means.
Renders two high-definition Indian Ocean heatmaps with:
  - Separated header / map / footer layout
  - DMI value in header strip
  - West + East pole boxes with value labels
  - Gridlines
  - Credit: Ankit Patel (bold blue) + gujaratweatherman.com (bold green)
  - Data source + climatology info in footer

Source: NOAA CoralTemp v3.1 (NCEI direct, 0.05° · 5km)
Climatology: 1985-1990 & 1993 (embedded in SSTA variable)

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
NCEI_CRW    = ("https://www.ncei.noaa.gov/data/oceans/crw/5km/v3.1"
               "/nc/v1.0/daily/ssta")
HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; IOD-monitor/1.0)"}

# Indian Ocean region
LAT_S_DEG, LAT_N_DEG = -30,  30
LON_W_DEG, LON_E_DEG =  30, 130

# Credits
AUTHOR  = "Ankit Patel"
WEBSITE = "gujaratweatherman.com"


# ── Data fetching ─────────────────────────────────────────────

def build_url(d):
    """Return CoralTemp SSTA URL for a given date."""
    yyyy = d.strftime("%Y")
    ymd  = d.strftime("%Y%m%d")
    return f"{NCEI_CRW}/{yyyy}/ct5km_ssta_v3.1_{ymd}.nc"


def download_file(d, tmpdir):
    """Download one daily CoralTemp SSTA file. Returns path or None."""
    url = build_url(d)
    out = os.path.join(tmpdir, f"coral_{d.isoformat()}.nc")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                f.write(chunk)
        size_mb = os.path.getsize(out) / 1024 / 1024
        print(f"  {d}  OK  ({size_mb:.1f} MB)")
        return out
    except Exception as e:
        print(f"  {d}  SKIP  ({e})")
        return None


def extract_ssta(nc_path):
    """
    Extract Indian Ocean SSTA from one CoralTemp NCEI file.
    Variable: sea_surface_temperature_anomaly
      dims:   (time, lat, lon)  — no zlev
      dtype:  int16
      scale:  0.01
      fill:   -32768
    Coords: lat / lon  (not latitude/longitude)
    Uses set_auto_maskandscale(False) for correct int16 handling.
    Returns 2D masked array (lat, lon) in °C.
    """
    ds   = nc.Dataset(nc_path)
    var  = ds.variables['sea_surface_temperature_anomaly']
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]

    # CoralTemp lats run 89.975 → -89.975 (N→S, reversed)
    # So index of -30S is LARGER than index of 30N — swap for slicing
    lat_n = int(np.argmin(np.abs(lats - LAT_N_DEG)))  # smaller index (north)
    lat_s = int(np.argmin(np.abs(lats - LAT_S_DEG)))  # larger index  (south)
    lon_w = int(np.argmin(np.abs(lons - LON_W_DEG)))
    lon_e = int(np.argmin(np.abs(lons - LON_E_DEG)))

    var.set_auto_maskandscale(False)
    raw   = var[0, lat_n:lat_s+1, lon_w:lon_e+1]   # (time, lat, lon) — no zlev
    scale = float(var.scale_factor)
    fill  = int(var._FillValue)                      # -32768
    ds.close()

    masked = np.ma.masked_where(raw == fill, raw)
    return masked.astype(np.float32) * scale


def get_lats_lons(nc_path):
    """Extract Indian Ocean lat/lon arrays from a CoralTemp NCEI file."""
    ds    = nc.Dataset(nc_path)
    lats  = ds.variables['lat'][:]    # 'lat' not 'latitude'
    lons  = ds.variables['lon'][:]    # 'lon' not 'longitude'
    # Lats are N→S (reversed): lat_n is smaller index, lat_s is larger
    lat_n = int(np.argmin(np.abs(lats - LAT_N_DEG)))
    lat_s = int(np.argmin(np.abs(lats - LAT_S_DEG)))
    lon_w = int(np.argmin(np.abs(lons - LON_W_DEG)))
    lon_e = int(np.argmin(np.abs(lons - LON_E_DEG)))
    result = (np.array(lats[lat_n:lat_s+1]),
              np.array(lons[lon_w:lon_e+1]))
    ds.close()
    return result


# ── Pole computation ──────────────────────────────────────────

def compute_poles(anom_2d, lats_sub, lons_sub):
    w_lat = np.where((lats_sub >= -10) & (lats_sub <= 10))[0]
    w_lon = np.where((lons_sub >=  50) & (lons_sub <= 70))[0]
    e_lat = np.where((lats_sub >= -10) & (lats_sub <=  0))[0]
    e_lon = np.where((lons_sub >=  90) & (lons_sub <= 110))[0]
    west  = round(float(np.ma.mean(anom_2d[np.ix_(w_lat, w_lon)])), 3)
    east  = round(float(np.ma.mean(anom_2d[np.ix_(e_lat, e_lon)])), 3)
    return west, east, round(west - east, 3)


# ── Map rendering ─────────────────────────────────────────────

def render_map(anom_2d, lats, lons, date_start, date_end,
               west, east, dmi, label, out_path):
    """
    Render one SST anomaly map with header/map/footer layout.
    Tries Cartopy for coastlines; falls back to plain Matplotlib.
    """
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    cmap    = plt.cm.RdBu_r
    vmax    = 2.0
    dmi_col = ('#f59e0b' if dmi >= 0.4
               else '#818cf8' if dmi <= -0.4
               else '#94a3b8')
    start_s = date_start.strftime("%d %b %Y")
    end_s   = date_end.strftime("%d %b %Y")

    # ── Figure with three separate axes ──────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0f1a', dpi=220)

    # Layout constants (figure fraction, bottom→top)
    HDR_B, HDR_H = 0.888, 0.112   # header bottom + height
    FTR_B, FTR_H = 0.000, 0.075   # footer bottom + height
    MAP_B = FTR_B + FTR_H + 0.005
    MAP_H = HDR_B - MAP_B - 0.005
    MAP_L, MAP_W = 0.055, 0.862   # left + width of map

    # ── HEADER ───────────────────────────────────────────────
    ax_h = fig.add_axes([0.0, HDR_B, 1.0, HDR_H])
    ax_h.set_facecolor('#0d1e30')
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1); ax_h.axis('off')
    ax_h.plot([0, 1], [0.04, 0.04], color='#1e3a52',
              lw=1.2, transform=ax_h.transAxes)

    ax_h.text(0.022, 0.68,
              f'Indian Ocean SST Anomaly — {label}',
              color='#e2e8f0', fontsize=14, fontweight='bold',
              va='center', fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.022, 0.22,
              f'{start_s}  →  {end_s}   ·   NOAA CoralTemp v3.1  ·  0.05° · 5km',
              color='#64748b', fontsize=9.5, va='center',
              fontfamily='monospace', transform=ax_h.transAxes)

    # DMI — right side, bold, value only
    ax_h.text(0.978, 0.50,
              f'DMI = {dmi:+.2f}°C',
              color=dmi_col, fontsize=18, fontweight='bold',
              va='center', ha='right', fontfamily='monospace',
              transform=ax_h.transAxes)

    # ── FOOTER ───────────────────────────────────────────────
    ax_f = fig.add_axes([0.0, FTR_B, 1.0, FTR_H])
    ax_f.set_facecolor('#0d1e30')
    ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1); ax_f.axis('off')
    ax_f.plot([0, 1], [0.94, 0.94], color='#1e3a52',
              lw=1.0, transform=ax_f.transAxes)

    # LEFT — credit
    ax_f.text(0.022, 0.46, 'Map generated by ',
              color='#94a3b8', fontsize=12, fontweight='normal',
              va='center', fontfamily='monospace',
              transform=ax_f.transAxes)
    ax_f.text(0.162, 0.46, AUTHOR,
              color='#38bdf8', fontsize=13, fontweight='bold',
              va='center', fontfamily='monospace',
              transform=ax_f.transAxes)

    # CENTRE — data source
    ax_f.text(0.50, 0.46,
              'Data: NOAA CoralTemp v3.1  ·  '
              'Anomaly vs 1985–1990 & 1993 climatology',
              color='#475569', fontsize=10, va='center',
              ha='center', fontfamily='monospace',
              transform=ax_f.transAxes)

    # RIGHT — website
    ax_f.text(0.978, 0.46, WEBSITE,
              color='#34d399', fontsize=13, fontweight='bold',
              va='center', ha='right', fontfamily='monospace',
              transform=ax_f.transAxes)

    # ── MAP ──────────────────────────────────────────────────
    rendered = False
    try:
        import cartopy.crs     as ccrs
        import cartopy.feature as cfeature

        ax = fig.add_axes([MAP_L, MAP_B, MAP_W, MAP_H],
                          projection=ccrs.PlateCarree())
        ax.set_extent([LON_W_DEG, LON_E_DEG, LAT_S_DEG, LAT_N_DEG],
                      crs=ccrs.PlateCarree())
        ax.set_facecolor('#0d1928')

        im = ax.pcolormesh(lons_2d, lats_2d, anom_2d,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='gouraud', zorder=1, rasterized=True)

        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'land', '110m',
            facecolor='#1a2b3c', edgecolor='none'), zorder=2)
        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'coastline', '110m',
            facecolor='none', edgecolor='#3d5a73',
            linewidth=0.8), zorder=3)

        # Gridlines — draw_labels=False avoids LinearRing crash
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     linewidth=0.5, color='#1e3a52', linestyle='--',
                     alpha=0.8, zorder=4,
                     xlocs=range(30, 131, 20),
                     ylocs=range(-30, 31, 10))

        ax.set_xticks(range(30, 131, 20), crs=ccrs.PlateCarree())
        ax.set_yticks(range(-30, 31, 10), crs=ccrs.PlateCarree())
        ax.set_xticklabels(
            [f'{x}°E' for x in range(30, 131, 20)],
            color='#64748b', fontsize=9, fontfamily='monospace')
        ax.set_yticklabels(
            [f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}'
             for y in range(-30, 31, 10)],
            color='#64748b', fontsize=9, fontfamily='monospace')
        ax.tick_params(color='#2d4a63', length=4)

        # Pole boxes
        ax.add_patch(mpatches.Rectangle(
            (50, -10), 20, 20, fill=False,
            edgecolor='#f59e0b', linewidth=2.2,
            transform=ccrs.PlateCarree(), zorder=6))
        ax.add_patch(mpatches.Rectangle(
            (90, -10), 20, 10, fill=False,
            edgecolor='#a78bfa', linewidth=2.2,
            transform=ccrs.PlateCarree(), zorder=6))

        # Pole labels
        for x, y, txt, col in [
            (60, 13.8, f'W: {west:+.2f}°C', '#f59e0b'),
            (100, 3.5, f'E: {east:+.2f}°C', '#a78bfa'),
        ]:
            ax.text(x, y, txt, color=col, fontsize=10.5,
                    fontweight='bold', ha='center', va='center',
                    zorder=8, transform=ccrs.PlateCarree(),
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#0a0f1a', alpha=0.88,
                              edgecolor=col, linewidth=0.9))

        for sp in ax.spines.values():
            sp.set_edgecolor('#1e3a52'); sp.set_linewidth(0.8)

        rendered = True
        print(f"  [{label}] Rendered with Cartopy")

    except Exception as e:
        print(f"  [{label}] Cartopy failed ({e}) — Matplotlib fallback")
        plt.close('all')
        fig = plt.figure(figsize=(16, 10), facecolor='#0a0f1a', dpi=220)
        # Re-draw header and footer in fallback figure
        _draw_header_footer(fig, HDR_B, HDR_H, FTR_B, FTR_H,
                            label, start_s, end_s, dmi, dmi_col)

    if not rendered:
        ax = fig.add_axes([MAP_L, MAP_B, MAP_W, MAP_H])
        ax.set_facecolor('#0d1928')
        im = ax.pcolormesh(lons_2d, lats_2d, anom_2d,
                           cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='gouraud', zorder=1, rasterized=True)
        ax.set_xlim(LON_W_DEG, LON_E_DEG)
        ax.set_ylim(LAT_S_DEG, LAT_N_DEG)

        for x in range(30, 131, 20):
            ax.axvline(x, color='#1e3a52', lw=0.5,
                       ls='--', alpha=0.8, zorder=4)
        for y in range(-30, 31, 10):
            ax.axhline(y, color='#1e3a52', lw=0.5,
                       ls='--', alpha=0.8, zorder=4)
        ax.axhline(0, color='#2a4a62', lw=0.9, zorder=4)

        ax.set_xticks(range(30, 131, 20))
        ax.set_yticks(range(-30, 31, 10))
        ax.set_xticklabels(
            [f'{x}°E' for x in range(30, 131, 20)],
            color='#64748b', fontsize=9, fontfamily='monospace')
        ax.set_yticklabels(
            [f'{abs(y)}°{"S" if y<0 else "N" if y>0 else ""}'
             for y in range(-30, 31, 10)],
            color='#64748b', fontsize=9, fontfamily='monospace')
        ax.tick_params(color='#2d4a63', length=4)

        ax.add_patch(mpatches.Rectangle(
            (50,-10), 20, 20, fill=False,
            edgecolor='#f59e0b', linewidth=2.2, zorder=6))
        ax.add_patch(mpatches.Rectangle(
            (90,-10), 20, 10, fill=False,
            edgecolor='#a78bfa', linewidth=2.2, zorder=6))

        for x, y, txt, col in [
            (60, 13.8, f'W: {west:+.2f}°C', '#f59e0b'),
            (100, 3.5, f'E: {east:+.2f}°C', '#a78bfa'),
        ]:
            ax.text(x, y, txt, color=col, fontsize=10.5,
                    fontweight='bold', ha='center', va='center',
                    zorder=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#0a0f1a', alpha=0.88,
                              edgecolor=col, linewidth=0.9))

        for sp in ax.spines.values():
            sp.set_edgecolor('#1e3a52'); sp.set_linewidth(0.8)

        print(f"  [{label}] Rendered with Matplotlib fallback")

    # ── Colorbar ─────────────────────────────────────────────
    cax  = fig.add_axes([MAP_L + MAP_W + 0.008,
                         MAP_B + 0.03,
                         0.015,
                         MAP_H - 0.06])
    cbar = plt.colorbar(im, cax=cax, extend='both')
    cbar.set_label('SST Anomaly (°C)', color='#94a3b8',
                   fontsize=9, labelpad=8, fontfamily='monospace')
    cbar.ax.yaxis.set_tick_params(color='#64748b', labelsize=8.5)
    plt.setp(cbar.ax.yaxis.get_ticklabels(),
             color='#64748b', fontfamily='monospace')
    cbar.outline.set_edgecolor('#1e3a52')
    cbar.ax.set_facecolor('#0a0f1a')

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=220, facecolor='#0a0f1a', edgecolor='none')
    plt.close()
    kb = os.path.getsize(out_path) / 1024
    print(f"  [{label}] Saved: {out_path}  ({kb:.0f} KB)")


def _draw_header_footer(fig, HDR_B, HDR_H, FTR_B, FTR_H,
                        label, start_s, end_s, dmi, dmi_col):
    """Re-draw header + footer on fallback figure."""
    ax_h = fig.add_axes([0.0, HDR_B, 1.0, HDR_H])
    ax_h.set_facecolor('#0d1e30')
    ax_h.set_xlim(0,1); ax_h.set_ylim(0,1); ax_h.axis('off')
    ax_h.plot([0,1],[0.04,0.04],color='#1e3a52',lw=1.2,
              transform=ax_h.transAxes)
    ax_h.text(0.022,0.68,f'Indian Ocean SST Anomaly — {label}',
              color='#e2e8f0',fontsize=14,fontweight='bold',
              va='center',fontfamily='monospace',
              transform=ax_h.transAxes)
    ax_h.text(0.022,0.22,
              f'{start_s}  →  {end_s}   ·   NOAA CoralTemp v3.1  ·  0.05° · 5km',
              color='#64748b',fontsize=9.5,va='center',
              fontfamily='monospace',transform=ax_h.transAxes)
    ax_h.text(0.978,0.50,f'DMI = {dmi:+.2f}°C',
              color=dmi_col,fontsize=18,fontweight='bold',
              va='center',ha='right',fontfamily='monospace',
              transform=ax_h.transAxes)

    ax_f = fig.add_axes([0.0, FTR_B, 1.0, FTR_H])
    ax_f.set_facecolor('#0d1e30')
    ax_f.set_xlim(0,1); ax_f.set_ylim(0,1); ax_f.axis('off')
    ax_f.plot([0,1],[0.94,0.94],color='#1e3a52',lw=1.0,
              transform=ax_f.transAxes)
    ax_f.text(0.022,0.46,'Map generated by ',
              color='#94a3b8',fontsize=12,va='center',
              fontfamily='monospace',transform=ax_f.transAxes)
    ax_f.text(0.162,0.46,AUTHOR,
              color='#38bdf8',fontsize=13,fontweight='bold',
              va='center',fontfamily='monospace',
              transform=ax_f.transAxes)
    ax_f.text(0.50,0.46,
              'Data: NOAA CoralTemp v3.1  ·  '
              'Anomaly vs 1985–1990 & 1993 climatology',
              color='#475569',fontsize=10,va='center',ha='center',
              fontfamily='monospace',transform=ax_f.transAxes)
    ax_f.text(0.978,0.46,WEBSITE,
              color='#34d399',fontsize=13,fontweight='bold',
              va='center',ha='right',fontfamily='monospace',
              transform=ax_f.transAxes)


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("IOD Pipeline — render_sst_map.py (CoralTemp v3.1)")
    print(f"Run time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    try:
        today      = date.today()
        # Start from 3 days ago — CoralTemp has ~4 day lag
        candidates = [today - timedelta(days=i) for i in range(3, 38)]

        print(f"Downloading up to 30 CoralTemp files "
              f"({candidates[0]} → {candidates[-1]})...")

        daily_anoms = []
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
                    if lats_sub is None:
                        lats_sub, lons_sub = get_lats_lons(path)
                    anom = extract_ssta(path)
                    daily_anoms.append(anom)
                    good_dates.append(d)
                except Exception as e:
                    print(f"  {d}  extract failed: {e}")

        if len(good_dates) < 7:
            raise RuntimeError(
                f"Only {len(good_dates)} files downloaded — need at least 7")

        good_dates  = sorted(good_dates)
        daily_anoms = list(reversed(daily_anoms))   # oldest → newest

        print(f"\nGot {len(good_dates)} files: "
              f"{good_dates[0]} → {good_dates[-1]}")

        # 7-day mean
        anom_7d  = np.ma.mean(np.ma.stack(daily_anoms[-7:]),  axis=0)
        dates_7d = good_dates[-7:]

        # 30-day mean
        anom_30d  = np.ma.mean(np.ma.stack(daily_anoms), axis=0)
        dates_30d = good_dates

        # Pole values
        w7,  e7,  d7  = compute_poles(anom_7d,  lats_sub, lons_sub)
        w30, e30, d30 = compute_poles(anom_30d, lats_sub, lons_sub)

        print(f"\nWeekly  — W:{w7:+.3f}  E:{e7:+.3f}  DMI:{d7:+.3f}°C")
        print(f"Monthly — W:{w30:+.3f}  E:{e30:+.3f}  DMI:{d30:+.3f}°C")

        # Render weekly
        render_map(
            anom_7d, lats_sub, lons_sub,
            dates_7d[0], dates_7d[-1],
            w7, e7, d7,
            label="7-Day Mean",
            out_path=os.path.join(OUTPUT_DIR, "sst_anomaly_weekly.png")
        )

        # Render monthly
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
            "source":        "NOAA CoralTemp v3.1 (NCEI, 0.05° · 5km)",
            "climatology":   "1985–1990 & 1993 baseline",
            "weekly": {
                "date_start":  dates_7d[0].isoformat(),
                "date_end":    dates_7d[-1].isoformat(),
                "n_days":      len(dates_7d),
                "west_pole":   {"region": "50–70°E, 10°S–10°N",
                                "anomaly_c": w7},
                "east_pole":   {"region": "90–110°E, 10°S–0°N",
                                "anomaly_c": e7},
                "derived_dmi": d7
            },
            "monthly": {
                "date_start":  dates_30d[0].isoformat(),
                "date_end":    dates_30d[-1].isoformat(),
                "n_days":      len(dates_30d),
                "west_pole":   {"region": "50–70°E, 10°S–10°N",
                                "anomaly_c": w30},
                "east_pole":   {"region": "90–110°E, 10°S–0°N",
                                "anomaly_c": e30},
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
