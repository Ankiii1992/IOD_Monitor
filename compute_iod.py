"""
compute_iod.py
Fetches IOD data from BOM (weekly observed) and JAMSTEC (forecast).
Outputs: output/iod_data.json

Sources:
  BOM     - weekly DMI, 2008-present, ~2 day lag
  JAMSTEC - 6-month ensemble forecast, monthly cadence

Run: python compute_iod.py
"""

import requests
import json
import os
from datetime import datetime, date

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IOD-monitor/1.0)"}
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "iod_data.json")


def get_phase(dmi):
    if   dmi >=  1.0: return "Strong Positive"
    elif dmi >=  0.4: return "Positive"
    elif dmi <= -1.0: return "Strong Negative"
    elif dmi <= -0.4: return "Negative"
    else:             return "Neutral"


# ── BOM Weekly DMI ───────────────────────────────────────────
def fetch_bom():
    print("Fetching BOM weekly DMI...")
    url = "https://www.bom.gov.au/clim_data/IDCK000072/iod_1.txt"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    records = []
    for line in r.text.strip().split('\n'):
        p = line.strip().split(',')
        if len(p) != 3:
            continue
        try:
            start = datetime.strptime(p[0], "%Y%m%d").date()
            end   = datetime.strptime(p[1], "%Y%m%d").date()
            dmi   = float(p[2])
            records.append({
                "start": start.isoformat(),
                "end":   end.isoformat(),
                "dmi":   round(dmi, 2)
            })
        except (ValueError, IndexError):
            continue

    latest = records[-1]
    lag    = (date.today() - date.fromisoformat(latest["end"])).days

    print(f"  BOM: {len(records)} weekly records | "
          f"latest week ending {latest['end']} | "
          f"DMI={latest['dmi']:+.2f} | lag={lag}d")

    return {
        "source":      "BOM",
        "description": "Bureau of Meteorology — HadISST based, weekly",
        "url":         url,
        "cadence":     "Weekly (updates Monday)",
        "baseline":    "1981–2010",
        "latest": {
            "week_ending": latest["end"],
            "dmi":         latest["dmi"],
            "phase":       get_phase(latest["dmi"]),
            "lag_days":    lag
        },
        "trend_4wk":   [r["dmi"] for r in records[-4:]],
        # Last 5 years of weekly data for the chart
        "weekly":      [
            {"date": r["end"], "dmi": r["dmi"]}
            for r in records
            if date.fromisoformat(r["end"]).year >= date.today().year - 5
        ]
    }


# ── JAMSTEC Forecast ─────────────────────────────────────────
def fetch_jamstec():
    print("Fetching JAMSTEC SINTEX-F forecast...")
    url = "https://www.jamstec.go.jp/virtualearth/data/SINTEX/SINTEX_DMI.csv"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    today    = date.today()
    forecast = []

    for line in r.text.strip().split('\n')[1:]:
        p = line.split(',')
        if len(p) < 3:
            continue
        try:
            t         = datetime.strptime(p[0].strip(), "%Y-%m-%d").date()
            mean_str  = p[2].strip()
            if mean_str and t > today:
                forecast.append({
                    "date":  t.isoformat(),
                    "dmi":   round(float(mean_str), 3),
                    "phase": get_phase(float(mean_str))
                })
        except (ValueError, IndexError):
            continue

    # Next 6 months only
    forecast = forecast[:6]
    print(f"  JAMSTEC: {len(forecast)} forecast months")
    if forecast:
        for f in forecast:
            print(f"    {f['date']}  {f['dmi']:+.3f}  [{f['phase']}]")

    return {
        "source":      "JAMSTEC",
        "description": "JAMSTEC SINTEX-F coupled model — ensemble mean",
        "url":         url,
        "cadence":     "Monthly (updated ~15th each month)",
        "note":        "Forecast only — not observed value",
        "forecast":    forecast
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("IOD Pipeline — compute_iod.py")
    print(f"Run time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 50)

    result = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "bom":      None,
        "jamstec":  None,
        "errors":   []
    }

    try:
        result["bom"] = fetch_bom()
    except Exception as e:
        msg = f"BOM fetch failed: {e}"
        print(f"  ERROR: {msg}")
        result["errors"].append(msg)

    try:
        result["jamstec"] = fetch_jamstec()
    except Exception as e:
        msg = f"JAMSTEC fetch failed: {e}"
        print(f"  ERROR: {msg}")
        result["errors"].append(msg)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nOutput written: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
