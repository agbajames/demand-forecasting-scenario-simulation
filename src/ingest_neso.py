"""Ingest NESO historic demand package metadata and annual CSV resources."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

try:
    from .utils import PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR, ensure_project_directories, save_json, safe_filename
except ImportError:  # Allows `python src/ingest_neso.py` from the project root.
    from utils import PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR, ensure_project_directories, save_json, safe_filename

API_URL = "https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data"
METADATA_PATH = RAW_DATA_DIR / "neso_package_metadata.json"
INVENTORY_PATH = TABLES_DIR / "neso_resource_inventory.csv"
SELECTED_RESOURCE_INFO_PATH = RAW_DATA_DIR / "selected_resource_info.json"
DEFAULT_START_YEAR = 2019
ANNUAL_RESOURCE_PATTERNS = [
    re.compile(r"historic[\W_]*demand[\W_]*data[\W_]*(20\d{2})", re.IGNORECASE),
    re.compile(r"demanddata[\W_]*(20\d{2})\.csv", re.IGNORECASE),
]


@dataclass(frozen=True)
class AnnualDemandResource:
    """A downloadable annual NESO historic demand CSV resource."""

    resource: dict[str, Any]
    year: int
    name: str
    url_or_path: str


def fetch_package_metadata(api_url: str = API_URL, timeout: int = 60) -> dict[str, Any]:
    """Fetch NESO package metadata from the CKAN API and return the package result."""
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(f"NESO API returned success=false: {payload}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("NESO API response did not contain a package result object.")
    return result


def extract_resources(package_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract resource dictionaries from package metadata safely."""
    resources = package_metadata.get("resources", [])
    return resources if isinstance(resources, list) else []


def extract_annual_year(resource: dict[str, Any]) -> int | None:
    """Extract the annual historic demand year from a resource name or URL."""
    name = str(resource.get("name") or resource.get("title") or "")
    url_or_path = str(resource.get("url") or resource.get("path") or "")
    for value in [name, url_or_path]:
        for pattern in ANNUAL_RESOURCE_PATTERNS:
            match = pattern.search(value)
            if match:
                return int(match.group(1))
    return None


def is_csv_resource(resource: dict[str, Any]) -> bool:
    """Return True when a resource is advertised as, or links to, a CSV."""
    fmt = str(resource.get("format") or resource.get("mimetype") or "").lower()
    url_or_path = str(resource.get("url") or resource.get("path") or "").lower().split("?")[0]
    return "csv" in fmt or url_or_path.endswith(".csv")


def identify_annual_historic_demand_resources(resources: list[dict[str, Any]]) -> list[AnnualDemandResource]:
    """Identify annual historic demand CSV resources and return them sorted by year."""
    annual_resources: list[AnnualDemandResource] = []
    seen_years: set[int] = set()
    for resource in resources:
        year = extract_annual_year(resource)
        url_or_path = str(resource.get("url") or resource.get("path") or "")
        if year is None or not url_or_path or not is_csv_resource(resource):
            continue
        if year in seen_years:
            continue
        annual_resources.append(
            AnnualDemandResource(
                resource=resource,
                year=year,
                name=str(resource.get("name") or resource.get("title") or f"historic_demand_data_{year}"),
                url_or_path=url_or_path,
            )
        )
        seen_years.add(year)
    return sorted(annual_resources, key=lambda item: item.year)


def build_resource_inventory(
    resources: list[dict[str, Any]], selected_years: set[int] | None = None
) -> pd.DataFrame:
    """Build a clean table describing available NESO resources."""
    selected_years = selected_years or set()
    rows: list[dict[str, Any]] = []
    for resource in resources:
        year = extract_annual_year(resource)
        rows.append(
            {
                "resource_name": resource.get("name") or resource.get("title"),
                "format": resource.get("format") or resource.get("mimetype"),
                "url_or_path": resource.get("url") or resource.get("path"),
                "resource_id": resource.get("id"),
                "last_modified": resource.get("last_modified") or resource.get("metadata_modified"),
                "datastore_available": resource.get("datastore_active"),
                "extracted_year": year,
                "selected_for_download": bool(year in selected_years),
            }
        )
    return pd.DataFrame(rows)


def latest_complete_year(available_years: list[int], include_partial_current_year: bool) -> int | None:
    """Return the latest available year allowed by the partial-current-year setting."""
    current_year = date.today().year
    allowed_years = [
        year
        for year in available_years
        if year < current_year or (include_partial_current_year and year == current_year)
    ]
    return max(allowed_years) if allowed_years else None


def select_annual_resources(
    annual_resources: list[AnnualDemandResource],
    start_year: int,
    end_year: int | None,
    include_partial_current_year: bool,
) -> list[AnnualDemandResource]:
    """Select annual resources within the requested complete-year range."""
    available_years = [item.year for item in annual_resources]
    effective_end_year = end_year or latest_complete_year(available_years, include_partial_current_year)
    if effective_end_year is None:
        return []
    current_year = date.today().year
    return [
        item
        for item in annual_resources
        if start_year <= item.year <= effective_end_year
        and (item.year < current_year or (include_partial_current_year and item.year == current_year))
    ]


def output_filename_from_resource(resource: dict[str, Any]) -> str:
    """Derive a stable local file name, preserving the remote file name where possible."""
    url = str(resource.get("url") or resource.get("path") or "")
    parsed_name = Path(unquote(urlparse(url).path)).name
    if parsed_name:
        return safe_filename(parsed_name)
    name = str(resource.get("name") or resource.get("id") or "neso_historic_demand.csv")
    suffix = ".csv" if not name.lower().endswith(".csv") else ""
    return f"{safe_filename(name)}{suffix}"


def download_resource(
    resource: dict[str, Any], output_dir: Path = RAW_DATA_DIR, timeout: int = 120, force: bool = False
) -> Path:
    """Download a selected NESO resource unless a non-empty local file already exists."""
    url = resource.get("url") or resource.get("path")
    if not url:
        raise ValueError("Selected resource does not include a URL/path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename_from_resource(resource)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"Skipping existing raw file: {output_path}")
        return output_path
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    with requests.get(str(url), stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {url}")
    temporary_path.replace(output_path)
    return output_path


def standardise_columns_for_combining(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names so annual files can be concatenated safely."""
    cleaned = df.copy()
    cleaned.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(column).strip().lower())).strip("_")
        for column in cleaned.columns
    ]
    return cleaned


def combine_annual_files(selected_files: list[tuple[AnnualDemandResource, Path]], output_path: Path) -> Path:
    """Concatenate selected annual CSV files and save a combined raw dataset."""
    frames: list[pd.DataFrame] = []
    for annual_resource, file_path in selected_files:
        frame = pd.read_csv(file_path)
        frame = standardise_columns_for_combining(frame)
        frame["source_year"] = annual_resource.year
        frames.append(frame)
    if not frames:
        raise ValueError("No annual files were available to combine.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output_path, index=False)
    return output_path


def relative_project_path(path: Path) -> str:
    """Return a path relative to the project root for portable JSON metadata."""
    return str(path.resolve().relative_to(PROJECT_ROOT))


def print_resource_summary(inventory: pd.DataFrame) -> None:
    """Print a readable console summary of available package resources."""
    print(f"Found {len(inventory)} NESO resource(s).")
    if inventory.empty:
        return
    for idx, row in inventory.iterrows():
        print(f"\n[{idx + 1}] {row.get('resource_name') or 'Unnamed resource'}")
        print(f"    Format: {row.get('format')}")
        print(f"    URL/path: {row.get('url_or_path')}")
        print(f"    Last modified: {row.get('last_modified')}")
        print(f"    Datastore available: {row.get('datastore_available')}")
        print(f"    Extracted year: {row.get('extracted_year')}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for selecting annual demand resources."""
    parser = argparse.ArgumentParser(description="Download annual NESO historic demand CSV files.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR, help="First annual file to download.")
    parser.add_argument("--end-year", type=int, default=None, help="Last annual file to download.")
    parser.add_argument(
        "--include-partial-current-year",
        action="store_true",
        help="Include the current year if NESO publishes it, even though it may be incomplete.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download annual raw files even when non-empty local copies already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch metadata, save inventory, download annual CSVs, and combine them."""
    args = parse_args()
    ensure_project_directories()
    try:
        package_metadata = fetch_package_metadata()
    except requests.HTTPError as exc:
        raise SystemExit(f"HTTP error while calling NESO API: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"Network error while calling NESO API: {exc}") from exc

    save_json(package_metadata, METADATA_PATH)
    resources = extract_resources(package_metadata)
    annual_resources = identify_annual_historic_demand_resources(resources)
    selected_resources = select_annual_resources(
        annual_resources=annual_resources,
        start_year=args.start_year,
        end_year=args.end_year,
        include_partial_current_year=args.include_partial_current_year,
    )
    selected_years = {resource.year for resource in selected_resources}
    inventory = build_resource_inventory(resources, selected_years=selected_years)
    inventory.to_csv(INVENTORY_PATH, index=False)
    print(f"Saved package metadata to {METADATA_PATH}")
    print(f"Saved resource inventory to {INVENTORY_PATH}")
    print_resource_summary(inventory)

    if not annual_resources:
        print("\nNo annual historic demand CSV resources could be identified. Manual selection is needed.")
        return
    if not selected_resources:
        print("\nNo annual historic demand CSV resources matched the requested year range.")
        return

    selected_start_year = min(selected_years)
    selected_end_year = max(selected_years)
    combined_output_path = RAW_DATA_DIR / f"neso_historic_demand_{selected_start_year}_{selected_end_year}.csv"
    print(
        f"\nSelected {len(selected_resources)} annual CSV resource(s): "
        f"{selected_start_year}-{selected_end_year}"
    )
    if not args.include_partial_current_year and date.today().year in [resource.year for resource in annual_resources]:
        print(f"Excluded partial current year {date.today().year}; pass --include-partial-current-year to include it.")

    downloaded_files: list[tuple[AnnualDemandResource, Path]] = []
    try:
        for annual_resource in selected_resources:
            downloaded_path = download_resource(annual_resource.resource, force=args.force_download)
            downloaded_files.append((annual_resource, downloaded_path))
        combine_annual_files(downloaded_files, combined_output_path)
    except requests.HTTPError as exc:
        raise SystemExit(f"HTTP error while downloading annual resources: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"Network error while downloading annual resources: {exc}") from exc

    selected_info = {
        "selected_year_range": {
            "start_year": selected_start_year,
            "end_year": selected_end_year,
            "requested_start_year": args.start_year,
            "requested_end_year": args.end_year,
        },
        "partial_current_year_included": args.include_partial_current_year,
        "annual_files_selected": len(selected_resources),
        "selected_resources": [
            {
                "name": annual_resource.name,
                "year": annual_resource.year,
                "url_or_path": annual_resource.url_or_path,
                "resource_id": annual_resource.resource.get("id"),
                "last_modified": annual_resource.resource.get("last_modified")
                or annual_resource.resource.get("metadata_modified"),
                "downloaded_path": relative_project_path(downloaded_path),
            }
            for annual_resource, downloaded_path in downloaded_files
        ],
        "combined_output_path": relative_project_path(combined_output_path),
        "selection_rationale": (
            "Selected annual CSV resources whose names or URLs match NESO historic demand data year patterns. "
            "By default the ingestion uses recent complete years from 2019 onward and excludes the current "
            "partial year unless --include-partial-current-year is supplied."
        ),
    }
    save_json(selected_info, SELECTED_RESOURCE_INFO_PATH)
    print(f"Saved combined raw dataset to {combined_output_path}")
    print(f"Saved selected resource documentation to {SELECTED_RESOURCE_INFO_PATH}")


if __name__ == "__main__":
    main()
