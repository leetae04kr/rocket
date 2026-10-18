import gc
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd

from pipeline.config import (
    DATASET_SCHEMA,
    OUTPUT_DIR,
    RAW_LL2_DIR
)
from pipeline.ll2_client import LL2DataCollector
from pipeline.gcat_client import GCATDataCollector
from pipeline.era5_client import ERA5BatchDownloader
from pipeline.joiner import SpatioTemporalJoiner

logger = logging.getLogger("RocketPipeline")


class RocketAtmospherePipeline:
    """
    End-to-End Orchestrator:
    1. Extracts mass historical orbital launch telemetry via GCAT (instant full archive) or LL2 API.
    2. Groups events temporally (Year-Month) for optimal ERA5 NetCDF batch querying.
    3. Executes spatio-temporal joins and vectorized atmospheric physics feature engineering.
    4. Writes memory-safe chunked CSV datasets with schema integrity.
    """

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.ll2_collector = LL2DataCollector()
        self.gcat_collector = GCATDataCollector()
        self.era5_downloader = ERA5BatchDownloader()
        self.joiner = SpatioTemporalJoiner()

    def run(
        self,
        max_launches: Optional[int] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        use_cached_ll2: bool = False,
        source: str = "gcat",
        combined_filename: str = "combined_rocket_weather_dataset.csv"
    ) -> Path:
        """Runs full ETL and ML dataset compilation pipeline."""
        logger.info(f"================ Starting Rocket Atmosphere Data Pipeline (Source: {source.upper()}) ================")

        # Step 1: Launch Telemetry Collection (GCAT instant archive or LL2 API)
        launches_df = self._obtain_telemetry_data(
            max_launches=max_launches,
            start_year=start_year,
            end_year=end_year,
            use_cached=use_cached_ll2,
            source=source
        )

        if launches_df.empty:
            logger.error("No launch records found. Exiting pipeline.")
            return self.output_dir / combined_filename

        # Convert timestamp and extract temporal indices
        launches_df["dt"] = pd.to_datetime(launches_df["launch_time_utc"])
        launches_df["year"] = launches_df["dt"].dt.year
        launches_df["month"] = launches_df["dt"].dt.month

        # Group by Year and Month for efficient ERA5 CDS batch requests
        groups = launches_df.groupby(["year", "month"])
        logger.info(f"Grouped {len(launches_df)} launches into {len(groups)} monthly batches for ERA5 ingestion.")

        combined_csv_path = self.output_dir / combined_filename
        if combined_csv_path.exists():
            combined_csv_path.unlink()  # Reset combined file for fresh execution

        total_processed_count = 0
        is_first_batch = True

        # Step 2 & 3: Iterate batches, download weather, join & stream to CSV
        for (year, month), batch_df in groups:
            logger.info(f"Processing Batch [{year}-{month:02d}]: {len(batch_df)} launches")

            days = sorted(batch_df["dt"].dt.strftime("%d").unique().tolist())
            hours = sorted(batch_df["dt"].dt.strftime("%H:00").unique().tolist())

            # ERA5 surface & pressure batches
            surf_path = self.era5_downloader.download_surface_batch(year=year, month=month, days=days, times=hours)
            press_path = self.era5_downloader.download_pressure_batch(year=year, month=month, days=days, times=hours)

            # Join
            joined_batch = self.joiner.join_launches_with_era5(
                launches_df=batch_df,
                surface_nc_path=surf_path,
                pressure_nc_path=press_path
            )

            if joined_batch.empty:
                continue

            # Ensure all schema columns exist
            for col in DATASET_SCHEMA:
                if col not in joined_batch.columns:
                    joined_batch[col] = None

            # Align strictly to target schema
            output_batch = joined_batch[DATASET_SCHEMA]

            # Save yearly chunk
            yearly_csv_path = self.output_dir / f"launches_weather_{year}.csv"
            write_header_yearly = not yearly_csv_path.exists()
            output_batch.to_csv(yearly_csv_path, mode="a", index=False, header=write_header_yearly)

            # Append to master combined CSV
            output_batch.to_csv(combined_csv_path, mode="a", index=False, header=is_first_batch)
            is_first_batch = False

            total_processed_count += len(output_batch)
            logger.info(f"Appended {len(output_batch)} records to {combined_csv_path} (Cumulative: {total_processed_count})")

            # Force memory cleanup
            del joined_batch
            del output_batch
            gc.collect()

        logger.info(f"Pipeline successfully generated master dataset at: {combined_csv_path}")
        self._print_dataset_summary(combined_csv_path)
        return combined_csv_path

    def _obtain_telemetry_data(
        self,
        max_launches: Optional[int],
        start_year: Optional[int],
        end_year: Optional[int],
        use_cached: bool,
        source: str = "gcat"
    ) -> pd.DataFrame:
        cache_file = RAW_LL2_DIR / f"{source}_launches_extracted.parquet"
        if use_cached and cache_file.exists():
            logger.info(f"Loading cached {source.upper()} data from {cache_file}")
            df = pd.read_parquet(cache_file)
            if start_year or end_year:
                df["dt_tmp"] = pd.to_datetime(df["launch_time_utc"])
                if start_year:
                    df = df[df["dt_tmp"].dt.year >= start_year]
                if end_year:
                    df = df[df["dt_tmp"].dt.year <= end_year]
                df = df.drop(columns=["dt_tmp"])
            if max_launches:
                df = df.head(max_launches)
            return df

        if source.lower() == "gcat":
            logger.info("Using GCAT (Jonathan McDowell Full Historical Catalog) collector...")
            return self.gcat_collector.collect_all_launches(
                max_records=max_launches,
                start_year=start_year,
                end_year=end_year
            )
        else:
            logger.info("Using LL2 (Launch Library 2) REST API collector...")
            return self.ll2_collector.collect_all_launches(
                max_records=max_launches,
                start_year=start_year,
                end_year=end_year
            )

    def _print_dataset_summary(self, csv_path: Path):
        """Displays data engineering statistics including class distribution."""
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        logger.info("================ Dataset Verification Summary ================")
        logger.info(f"Total Rows: {len(df)}")
        logger.info(f"Columns ({len(df.columns)}): {list(df.columns)}")
        if "launch_status" in df.columns:
            status_counts = df["launch_status"].value_counts().to_dict()
            logger.info(f"Target Distribution (launch_status): {status_counts}")
            success_rate = (status_counts.get(1, 0) / len(df)) * 100 if len(df) > 0 else 0
            logger.info(f"Overall Success Rate: {success_rate:.2f}%")
        logger.info("Missing Value Ratio:")
        nulls = df.isnull().mean()
        for col, null_pct in nulls[nulls > 0].items():
            logger.info(f"  - {col}: {null_pct * 100:.1f}%")
        logger.info("==============================================================")
