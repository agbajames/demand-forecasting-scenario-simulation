"""Ingest NESO historic demand package metadata and a suitable raw resource."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

try:
    from .utils import RAW_DATA_DIR, TABLES_DIR, ensure_project_directories, save_json, safe_filename
except ImportError:  # Allows `python src/ingest_neso.py` from the project root.
    from utils import RAW_DATA_DIR, TABLES_DIR, ensure_project_directories, save_json, safe_filename

API_URL = "https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data"
METADATA_PATH = RAW_DATA_DIR / "neso_package_metadata.json"
INVENTORY_PATH = TABLES_DIR / "neso_resource_inventory.csv"
SELECTED_RESOURCE_INFO_PATH = RAW_DATA_DIR / "selected_resource_info.json"


@dataclass(frozen=True)
class SelectedResource:
    """A downloadable resource chosen from the NESO data package."""

    resource: dict[str, Any]
    score: int
    reason: str


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


def build_resource_inventory(resources: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a clean table describing available NESO resources."""
    rows: list[dict[str, Any]] = []
    for resource in resources:
        rows.append(
            {
                "resource_name": resource.get("name") or resource.get("title"),
                "description": resource.get("description"),
                "format": resource.get("format") or resource.get("mimetype"),
                "url_or_path": resource.get("url") or resource.get("path"),
                "resource_id": resource.get("id"),
                "last_modified": resource.get("last_modified") or resource.get("metadata_modified"),
                "size": resource.get("size"),
                "datastore_available": resource.get("datastore_active"),
            }
        )
    return pd.DataFrame(rows)


def score_resource(resource: dict[str, Any]) -> tuple[int, list[str]]:
    """Score a package resource for relevance to raw historic electricity demand observations."""
    name = str(resource.get("name") or "")
    description = str(resource.get("description") or "")
    url = str(resource.get("url") or resource.get("path") or "")
    fmt = str(resource.get("format") or "")
    haystack = " ".join([name, description, url, fmt]).lower()
    score = 0
    reasons: list[str] = []

    if "csv" in fmt.lower() or url.lower().split("?")[0].endswith(".csv"):
        score += 50
        reasons.append("CSV resource")
    if "demand" in haystack:
        score += 30
        reasons.append("mentions demand")
    for term in ["historic", "historical", "history"]:
        if term in haystack:
            score += 15
            reasons.append(f"mentions {term}")
            break
    observation_terms = ["settlement", "half", "hour", "national", "transmission", "actual", "tsd", "nd"]
    matches = [term for term in observation_terms if term in haystack]
    score += min(len(matches), 5) * 5
    if matches:
        reasons.append(f"observation-like terms: {', '.join(matches[:5])}")
    metadata_terms = ["metadata", "schema", "dictionary", "readme", "documentation"]
    if any(term in haystack for term in metadata_terms):
        score -= 40
        reasons.append("penalised metadata/documentation terms")
    if not url:
        score -= 100
        reasons.append("no URL/path available")
    return score, reasons


def select_resource(resources: list[dict[str, Any]], minimum_score: int = 70) -> SelectedResource | None:
    """Select the most relevant CSV resource, or return None if confidence is too low."""
    candidates = []
    for resource in resources:
        score, reasons = score_resource(resource)
        candidates.append(SelectedResource(resource=resource, score=score, reason="; ".join(reasons)))
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: item.score)
    return selected if selected.score >= minimum_score else None


def output_filename_from_resource(resource: dict[str, Any]) -> str:
    """Derive a stable local file name, preserving the remote file name where possible."""
    url = str(resource.get("url") or resource.get("path") or "")
    parsed_name = Path(unquote(urlparse(url).path)).name
    if parsed_name:
        return safe_filename(parsed_name)
    name = str(resource.get("name") or resource.get("id") or "neso_historic_demand.csv")
    suffix = ".csv" if not name.lower().endswith(".csv") else ""
    return f"{safe_filename(name)}{suffix}"


def download_resource(resource: dict[str, Any], output_dir: Path = RAW_DATA_DIR, timeout: int = 120) -> Path:
    """Download a selected NESO resource to the raw data directory without altering contents."""
    url = resource.get("url") or resource.get("path")
    if not url:
        raise ValueError("Selected resource does not include a URL/path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename_from_resource(resource)
    with requests.get(str(url), stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return output_path


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


def main() -> None:
    """Fetch metadata, save inventory, and download a suitable raw NESO demand resource."""
    ensure_project_directories()
    try:
        package_metadata = fetch_package_metadata()
    except requests.HTTPError as exc:
        raise SystemExit(f"HTTP error while calling NESO API: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"Network error while calling NESO API: {exc}") from exc

    save_json(package_metadata, METADATA_PATH)
    resources = extract_resources(package_metadata)
    inventory = build_resource_inventory(resources)
    inventory.to_csv(INVENTORY_PATH, index=False)
    print(f"Saved package metadata to {METADATA_PATH}")
    print(f"Saved resource inventory to {INVENTORY_PATH}")
    print_resource_summary(inventory)

    selected = select_resource(resources)
    if selected is None:
        print("\nNo suitable CSV demand observation resource could be selected confidently. Manual selection is needed.")
        return

    print(f"\nSelected resource: {selected.resource.get('name')} (score={selected.score})")
    print(f"Reason: {selected.reason}")
    selected_info = {"selected_resource": selected.resource, "score": selected.score, "selection_reason": selected.reason}
    try:
        downloaded_path = download_resource(selected.resource)
    except requests.HTTPError as exc:
        raise SystemExit(f"HTTP error while downloading selected resource: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"Network error while downloading selected resource: {exc}") from exc
    selected_info["downloaded_path"] = str(downloaded_path.relative_to(RAW_DATA_DIR.parents[1]))
    save_json(selected_info, SELECTED_RESOURCE_INFO_PATH)
    print(f"Downloaded selected raw resource to {downloaded_path}")
    print(f"Saved selected resource documentation to {SELECTED_RESOURCE_INFO_PATH}")


if __name__ == "__main__":
    main()
