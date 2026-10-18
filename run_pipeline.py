#!/usr/bin/env python3
"""
CLI entry point for the Rocket Atmospheric Data Pipeline.
"""
import os
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.pipeline import RocketAtmospherePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Mass Extraction Pipeline: LL2 Orbital Telemetry + ECMWF ERA5 Atmospheric Reanalysis Join"
    )
    parser.add_argument(
        "--max-launches",
        type=int,
        default=None,
        help="Maximum number of launch events to collect (default: None, extracts all)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Filter launches from this year onwards (e.g. 2015)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Filter launches up to this year (e.g. 2024)"
    )
    parser.add_argument(
        "--use-cached-ll2",
        action="store_true",
        help="Skip LL2 API calls if cached parquet file exists"
    )
    parser.add_argument(
        "--dev-endpoint",
        action="store_true",
        help="Use LL2 Developer endpoint (lldev.thespacedevs.com) with higher rate limit allowance"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gcat",
        choices=["gcat", "ll2"],
        help="Telemetry data source: 'gcat' (Jonathan McDowell instant full historical catalog, 1957-2024, recommended) or 'll2' (Launch Library 2 API)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="combined_rocket_weather_dataset.csv",
        help="Name of the final combined CSV dataset"
    )

    args = parser.parse_args()

    if args.dev_endpoint:
        os.environ["LL2_BASE_URL"] = "https://lldev.thespacedevs.com/2.2.0/launch/previous/"
        print("[Config] Using LL2 Developer Endpoint.")

    pipeline = RocketAtmospherePipeline()
    output_path = pipeline.run(
        max_launches=args.max_launches,
        start_year=args.start_year,
        end_year=args.end_year,
        use_cached_ll2=args.use_cached_ll2,
        source=args.source,
        combined_filename=args.output_file
    )

    print(f"\n[Success] Dataset compilation complete. Output file: {output_path}")


if __name__ == "__main__":
    main()
