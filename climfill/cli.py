#!/usr/bin/env python3
"""
ClimFill CLI — Command-line interface.

Usage:
    python -m climfill.cli input.csv --output filled.csv
    climfill input.csv --method auto --uncertainty --lat 10.0 --lon 79.0
"""

import argparse
import sys
import json

from .engine import ClimFillEngine


def main():
    parser = argparse.ArgumentParser(
        description="ClimFill — Automated Climate Station Data Gap-Filler",
    )
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument("--date-col", default=None, help="Date column name")
    parser.add_argument(
        "--method",
        choices=["auto", "linear", "spline", "seasonal", "random_forest", "xgboost"],
        default="auto",
    )
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument(
        "--reanalysis",
        action="store_true",
        help="Download Open-Meteo reanalysis (requires --lat and --lon)",
    )
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--bootstrap-n", type=int, default=50)
    parser.add_argument("--no-flags", action="store_true")
    parser.add_argument("--report", default=None, help="JSON report path")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.reanalysis and (args.lat is None or args.lon is None):
        parser.error("--reanalysis requires --lat and --lon")

    def log(msg):
        if not args.quiet:
            print(msg)

    log("🌍 ClimFill — Automated Climate Station Data Gap-Filler\n")

    engine = ClimFillEngine()

    # Load
    log(f"📤 Loading {args.input}...")
    info = engine.load_csv(args.input, date_col=args.date_col)
    log(f"   {info['shape'][0]} records, {info['shape'][1]} columns")
    log(f"   Date range: {info['date_range']}")

    # Detect gaps
    log("\n🔍 Detecting gaps...")
    result = engine.detect_gaps()
    for col, gr in result["gap_report"].items():
        if gr["missing_count"] > 0:
            log(
                f"   {col}: {gr['missing_count']} missing "
                f"({gr['missing_pct']}%), max gap {gr['max_gap_length']}"
            )

    total_missing = sum(
        gr["missing_count"] for gr in result["gap_report"].values()
    )
    if total_missing == 0:
        log("\n✅ No gaps detected!")
        sys.exit(0)

    # Reanalysis
    reanalysis_map = {}
    if args.reanalysis and args.lat is not None and args.lon is not None:
        log(f"\n🌐 Downloading reanalysis for ({args.lat}, {args.lon})...")
        try:
            from .reanalysis import get_open_meteo_data, suggest_reanalysis_mapping

            dates = engine.df[engine.date_col]
            rean = get_open_meteo_data(
                lat=args.lat,
                lon=args.lon,
                start_date=dates.min().strftime("%Y-%m-%d"),
                end_date=dates.max().strftime("%Y-%m-%d"),
            )
            rean_cols = engine.merge_reanalysis(rean)
            reanalysis_map = suggest_reanalysis_mapping(
                engine.station_columns(), rean_cols
            )
            log(f"   Downloaded {len(rean)} days")
            for col, rcol in reanalysis_map.items():
                log(f"   {col} ← {rcol}")
        except Exception as e:
            log(f"   ⚠️ Failed: {e}. Continuing without reanalysis.")

    # Fill
    log(f"\n🔧 Filling gaps ({args.method})...")
    engine.fill_gaps(method=args.method, reanalysis_map=reanalysis_map)

    # Uncertainty
    if args.uncertainty:
        log(f"\n📊 Uncertainty ({args.bootstrap_n} bootstrap)...")
        engine.compute_uncertainty(n_bootstrap=args.bootstrap_n)

    # Export
    output_path = args.output or args.input.replace(".csv", "_filled.csv")
    log(f"\n📥 Exporting to {output_path}...")
    export_df = engine.export(
        filepath=output_path,
        include_uncertainty=args.uncertainty,
        include_metadata=not args.no_flags,
    )

    # Summary
    report = engine.generate_report()
    log("\n" + "=" * 50)
    for col, gs in report.get("gap_summary", {}).items():
        log(
            f"  {col}: {gs['filled_count']}/{gs['original_missing']} "
            f"({gs['fill_rate_pct']}%)"
        )

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"  Report: {args.report}")

    log("\n🎉 Done!")


if __name__ == "__main__":
    main()
