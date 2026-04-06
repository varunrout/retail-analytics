"""
Bronze layer loader for HealthBeauty360.

Loads raw JSON / Parquet files into the bronze layer.

* ``USE_GCP=false`` (default) — processes files locally and writes enriched
  parquet files to ``data/bronze/``.
* ``USE_GCP=true`` — additionally loads each table into a BigQuery bronze
  dataset.  Requires ``GOOGLE_CLOUD_PROJECT`` and ``BQ_BRONZE_DATASET``
  environment variables.

Usage::

    python -m raw_to_bronze.bronze_loader

or as a library::

    from raw_to_bronze.bronze_loader import BronzeLoader
    loader = BronzeLoader()
    counts = loader.load_all_raw()
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_RAW_DIR = _REPO_ROOT / "data" / "raw"
_BRONZE_DIR = _REPO_ROOT / "data" / "bronze"


def _bronze_dir() -> Path:
    """Return the local bronze output directory, creating it if needed."""
    _BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    return _BRONZE_DIR


# ---------------------------------------------------------------------------
# BronzeLoader
# ---------------------------------------------------------------------------

class BronzeLoader:
    """Loads raw data files into the bronze layer.

    Parameters
    ----------
    use_gcp:
        When ``True``, tables are also written to BigQuery.  Requires
        the ``google-cloud-bigquery`` package and valid ADC credentials.
        Defaults to the ``USE_GCP`` environment variable (``false``).
    """

    def __init__(self, use_gcp: bool | None = None) -> None:
        if use_gcp is None:
            use_gcp = os.environ.get("USE_GCP", "false").lower() in ("true", "1", "yes")
        self.use_gcp: bool = use_gcp

        if self.use_gcp:
            self._bq_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            self._bq_dataset = os.environ.get("BQ_BRONZE_DATASET", "bronze")
            if not self._bq_project:
                raise EnvironmentError(
                    "USE_GCP=true but GOOGLE_CLOUD_PROJECT environment variable is not set."
                )
            try:
                from google.cloud import bigquery  # noqa: F401
                self._bq_client = bigquery.Client(project=self._bq_project)
                logger.info(
                    "BigQuery client initialised for project=%s dataset=%s",
                    self._bq_project,
                    self._bq_dataset,
                )
            except ImportError as exc:
                raise ImportError(
                    "google-cloud-bigquery is required when USE_GCP=true. "
                    "Install it with: pip install google-cloud-bigquery"
                ) from exc

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_ndjson(
        self,
        source_path: Path,
        table_name: str,
        source_system: str = "raw",
    ) -> pd.DataFrame:
        """Load a newline-delimited JSON file and write it to the bronze layer.

        Parameters
        ----------
        source_path:
            Path to the ``.ndjson`` / ``.json`` source file.
        table_name:
            Logical table name used for the bronze output file.
        source_system:
            Label stored in the ``_source_system`` metadata column.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame as written to the bronze layer.
        """
        logger.info("load_ndjson: %s → %s", source_path.name, table_name)
        try:
            df = pd.read_json(source_path, lines=True)
        except ValueError:
            # Fall back to standard JSON array / object
            with source_path.open() as fh:
                raw: Any = json.load(fh)
            if isinstance(raw, list):
                df = pd.json_normalize(raw)
            elif isinstance(raw, dict):
                # Try common wrapper keys; otherwise flatten top-level values
                for key in ("results", "data", "records", "items"):
                    if key in raw and isinstance(raw[key], list):
                        df = pd.json_normalize(raw[key])
                        break
                else:
                    # Treat each top-level key as a row (e.g. bank_holidays)
                    rows = []
                    for k, v in raw.items():
                        if isinstance(v, list):
                            for item in v:
                                item["_region"] = k
                                rows.append(item)
                        else:
                            rows.append({"_key": k, "_value": v})
                    df = pd.json_normalize(rows) if rows else pd.DataFrame()
            else:
                df = pd.DataFrame([{"value": raw}])

        df = self._add_metadata(df, source_file=source_path.name, source_system=source_system)
        return self._persist(df, table_name)

    def load_parquet(
        self,
        source_path: Path,
        table_name: str,
        source_system: str = "raw",
    ) -> pd.DataFrame:
        """Load a Parquet file and write it to the bronze layer.

        Parameters
        ----------
        source_path:
            Path to the source ``.parquet`` file.
        table_name:
            Logical table name used for the bronze output file.
        source_system:
            Label stored in the ``_source_system`` metadata column.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame as written to the bronze layer.
        """
        logger.info("load_parquet: %s → %s", source_path.name, table_name)
        df = pd.read_parquet(source_path, engine="pyarrow")
        df = self._add_metadata(df, source_file=source_path.name, source_system=source_system)
        return self._persist(df, table_name)

    def load_csv(
        self,
        source_path: Path,
        table_name: str,
        source_system: str = "raw",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a CSV file and write it to the bronze layer.

        Parameters
        ----------
        source_path:
            Path to the source ``.csv`` file.
        table_name:
            Logical table name used for the bronze output file.
        source_system:
            Label stored in the ``_source_system`` metadata column.
        **kwargs:
            Extra keyword arguments forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame as written to the bronze layer.
        """
        logger.info("load_csv: %s → %s", source_path.name, table_name)
        df = pd.read_csv(source_path, **kwargs)
        df = self._add_metadata(df, source_file=source_path.name, source_system=source_system)
        return self._persist(df, table_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_metadata(
        self,
        df: pd.DataFrame,
        source_file: str,
        source_system: str,
    ) -> pd.DataFrame:
        """Append standard bronze metadata columns to *df*.

        Added columns
        -------------
        ``_ingested_at``
            UTC timestamp at time of ingestion.
        ``_source_file``
            Original filename.
        ``_source_system``
            Logical system label (e.g. ``"uci_retail"``).
        ``_row_hash``
            MD5 hex-digest of concatenated string representation of every
            field value, used for lightweight deduplication downstream.

        Parameters
        ----------
        df:
            Input DataFrame.
        source_file:
            Filename to store.
        source_system:
            System label to store.

        Returns
        -------
        pd.DataFrame
            DataFrame with four additional metadata columns appended.
        """
        now = datetime.now(tz=timezone.utc)
        df = df.copy()
        df["_ingested_at"] = now
        df["_source_file"] = source_file
        df["_source_system"] = source_system

        # Row hash: MD5 of pipe-joined string representation of all *original* fields.
        # Convert every cell to str explicitly to handle NaN, nested objects, etc.
        base_cols = df.drop(
            columns=["_ingested_at", "_source_file", "_source_system"], errors="ignore"
        )
        row_strings = base_cols.apply(
            lambda row: "|".join(str(v) for v in row.values), axis=1
        )
        df["_row_hash"] = row_strings.apply(
            lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()  # noqa: S324
        )
        return df

    def _persist(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Save *df* locally and, if GCP is enabled, to BigQuery."""
        self._save_local(df, table_name)
        if self.use_gcp:
            self._save_to_bq(df, table_name)
        return df

    def _save_local(self, df: pd.DataFrame, table_name: str) -> Path:
        """Write *df* to ``data/bronze/{table_name}.parquet``.

        Parameters
        ----------
        df:
            DataFrame to persist.
        table_name:
            Used as the output filename stem.

        Returns
        -------
        Path
            Path of the written parquet file.
        """
        out_path = _bronze_dir() / f"{table_name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Saved bronze/%s.parquet (%d rows).", table_name, len(df))
        return out_path

    def _save_to_bq(self, df: pd.DataFrame, table_name: str) -> str:
        """Write *df* to BigQuery bronze dataset.

        The table is written using ``WRITE_TRUNCATE`` — full refresh on
        each pipeline run.  For incremental loads, override this method.

        Parameters
        ----------
        df:
            DataFrame to upload.
        table_name:
            BigQuery table name within the bronze dataset.

        Returns
        -------
        str
            Fully-qualified BigQuery table ID
            (``project.dataset.table_name``).
        """
        from google.cloud import bigquery  # noqa: PLC0415

        table_id = f"{self._bq_project}.{self._bq_dataset}.{table_name}"
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        )
        # Convert timezone-aware timestamps to UTC-naive (BQ requirement)
        df_upload = df.copy()
        for col in df_upload.select_dtypes(include=["datetimetz"]).columns:
            df_upload[col] = df_upload[col].dt.tz_convert("UTC").dt.tz_localize(None)

        job = self._bq_client.load_table_from_dataframe(
            df_upload, table_id, job_config=job_config
        )
        job.result()  # Wait for completion
        logger.info("Loaded %d rows to BigQuery table %s.", len(df), table_id)
        return table_id

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def load_all_raw(self) -> dict[str, int]:
        """Load all known raw files into the bronze layer.

        Iterates over a fixed manifest of raw sources.  Missing files are
        logged as warnings rather than raising exceptions so that a partial
        raw directory still produces useful bronze output.

        Returns
        -------
        dict[str, int]
            Mapping of ``table_name → row_count`` for every successfully
            loaded table.
        """
        counts: dict[str, int] = {}

        # ------------------------------------------------------------------
        # Manifest: (table_name, loader_method, relative_raw_path, kwargs)
        # ------------------------------------------------------------------
        manifest: list[tuple[str, str, str, dict]] = [
            (
                "bank_holidays",
                "ndjson",
                "bank_holidays/bank_holidays.json",
                {},
            ),
            (
                "weather",
                "ndjson",
                # Load all monthly weather JSON files from the london subfolder
                "weather/london",
                {},
            ),
            (
                "ons_retail_sales",
                "ndjson",
                "ons/ons_retail_sales.json",
                {},
            ),
            (
                "ons_internet_sales",
                "ndjson",
                "ons/ons_internet_sales.json",
                {},
            ),
            (
                "google_trends",
                "ndjson",
                "google_trends/trends.json",
                {},
            ),
            (
                "beauty_products",
                "ndjson",
                "open_beauty_facts/products.json",
                {},
            ),
            (
                "shopify_products",
                "parquet",
                "shopify/products.parquet",
                {},
            ),
            (
                "ebay_listings",
                "parquet",
                "ebay/listings.parquet",
                {},
            ),
            (
                "trade_data",
                "parquet",
                "trade/uk_trade_data.parquet",
                {},
            ),
        ]

        # UCI retail parquet files (one per year)
        for year in range(2020, 2025):
            uci_path = f"uci_retail/{year}.parquet"
            if (_RAW_DIR / uci_path).exists():
                manifest.append(
                    (
                        f"uci_transactions_{year}",
                        "parquet",
                        uci_path,
                        {},
                    )
                )

        for table_name, loader_type, rel_path, kwargs in manifest:
            src = _RAW_DIR / rel_path
            try:
                if loader_type == "ndjson":
                    if src.is_dir():
                        # Load all JSON files in directory and concatenate
                        frames = []
                        for jf in sorted(src.glob("*.json")):
                            try:
                                frames.append(
                                    self.load_ndjson(
                                        jf,
                                        table_name=f"{table_name}_{jf.stem}",
                                        source_system=table_name,
                                    )
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Skipping %s: %s", jf.name, exc
                                )
                        if frames:
                            combined = pd.concat(frames, ignore_index=True)
                            # Save a combined version too
                            self._save_local(combined, table_name)
                            counts[table_name] = len(combined)
                        else:
                            logger.warning("No JSON files found in %s", src)
                        continue
                    if not src.exists():
                        logger.warning("Raw file not found, skipping: %s", src)
                        continue
                    df = self.load_ndjson(src, table_name, source_system=table_name)
                elif loader_type == "parquet":
                    if not src.exists():
                        logger.warning("Raw file not found, skipping: %s", src)
                        continue
                    df = self.load_parquet(src, table_name, source_system=table_name)
                elif loader_type == "csv":
                    if not src.exists():
                        logger.warning("Raw file not found, skipping: %s", src)
                        continue
                    df = self.load_csv(src, table_name, source_system=table_name, **kwargs)
                else:
                    logger.error("Unknown loader type '%s' for %s.", loader_type, table_name)
                    continue

                counts[table_name] = len(df)

            except Exception as exc:
                logger.exception("Failed to load '%s' from %s: %s", table_name, rel_path, exc)

        logger.info(
            "Bronze loading complete. Tables loaded: %d | Total rows: %d",
            len(counts),
            sum(counts.values()),
        )
        return counts


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def run() -> dict[str, int]:
    """Instantiate a :class:`BronzeLoader` and load all raw files.

    Returns
    -------
    dict[str, int]
        Table-name to row-count mapping.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    loader = BronzeLoader()
    counts = loader.load_all_raw()
    print("\nBronze layer summary:")
    print(f"  {'Table':<35} {'Rows':>8}")
    print("  " + "-" * 44)
    for tbl, cnt in sorted(counts.items()):
        print(f"  {tbl:<35} {cnt:>8,}")
    print(f"\n  Total tables: {len(counts)}  |  Total rows: {sum(counts.values()):,}")
    return counts


if __name__ == "__main__":
    run()
