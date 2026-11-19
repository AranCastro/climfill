#!/usr/bin/env python3
"""
ClimFill CLI — Command-line interface.

Usage:
    python -m climfill.cli input.csv --output filled.csv
    climfill input.csv --method auto --uncertainty --lat 10.0 --lon 79.0
"""

import argparse
import json
from pathlib import Path

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
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Add 95%% intervals derived from hold-out residuals",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=None,
        help="Deprecated; ignored (intervals no longer use a bootstrap)",
    )
    parser.add_argument(
        "--sentinels",
        default=None,
        help="Comma-separated missing-value codes (default: -999,-9999,-99.9)",
    )
    parser.add_argument(
        "--no-regularise",
        action="store_true",
        help="Do not insert rows for timestamps absent from the file",
    )
    parser.add_argument("--no-flags", action="store_true")
    parser.add_argument("--report", default=None, help="JSON report path")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.reanalysis and (args.lat is None or args.lon is None):
        parser.error("--reanalysis requires --lat and --lon")

    sentinels = None
    if args.sentinels is not None:
        try:
            sentinels = [float(v) for v in args.sentinels.split(",") if v.strip()]
        except ValueError:
            parser.error("--sentinels expects comma-separated numbers")

    # Never write over the observations: the default name is derived from
    # the stem, whatever the extension
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_filled{input_path.suffix or '.csv'}"
    )
    if output_path.resolve() == input_path.resolve():
        parser.error("output path is the input file; choose another --output")

    def log(msg):
        if not args.quiet:
            print(msg)

    log("🌍 ClimFill — Automated Climate Station Data Gap-Filler\n")

    engine = ClimFillEngine()

    # Load
    log(f"📤 Loading {args.input}...")
    info = engine.load_csv(
        args.input,
        date_col=args.date_col,
        sentinels=sentinels,
        regularise=not args.no_regularise,
    )
    log(f"   {info['shape'][0]} records, {info['shape'][1]} columns")
    log(f"   Date range: {info['date_range']}")
    if info["rows_inserted"]:
        log(f"   {info['rows_inserted']} absent timestamps inserted as gaps")

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
        log("\n✅ No gaps detected; the output is a copy of the input data.")

    # Reanalysis
    reanalysis_map = {}
    if total_missing and args.reanalysis:
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
        log("\n📊 Uncertainty (hold-out residual intervals)...")
        if args.bootstrap_n is not None:
            log("   --bootstrap-n is deprecated and ignored")
        engine.compute_uncertainty()

    # Export
    log(f"\n📥 Exporting to {output_path}...")
    engine.export(
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
