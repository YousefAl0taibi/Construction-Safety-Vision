from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


USER_AGENT = (
    "Construction-Safety-Vision/0.1.0 "
    "(research dataset curation; source links are recorded in data/saudi/sources.csv)"
)
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
RETRIABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class SocialImageParser(HTMLParser):
    """Extract the public preview image declared by an asset page."""

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {key.lower(): value for key, value in attrs if value is not None}
        key = (values.get("property") or values.get("name") or "").lower()
        if key in {"og:image:secure_url", "og:image", "twitter:image"}:
            content = values.get("content")
            if content:
                self.candidates.append(content)


def request_bytes(
    url: str, max_bytes: int = MAX_DOWNLOAD_BYTES, retries: int = 2
) -> tuple[bytes, str, str]:
    """Fetch a bounded response, retrying temporary server and rate-limit errors."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=45) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError(f"remote file is larger than {max_bytes} bytes")

                chunks: list[bytes] = []
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"download exceeded {max_bytes} bytes")
                    chunks.append(chunk)
                return (
                    b"".join(chunks),
                    response.headers.get_content_type(),
                    response.geturl(),
                )
        except HTTPError as error:
            if error.code not in RETRIABLE_HTTP_STATUS or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(min(delay, 8.0))
        except (URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def resolve_asset_url(source_page: str, explicit_asset_url: str = "") -> str:
    if explicit_asset_url:
        return explicit_asset_url

    page, content_type, final_page_url = request_bytes(source_page, max_bytes=4 * 1024 * 1024)
    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(f"expected an HTML source page, received {content_type}")

    parser = SocialImageParser()
    parser.feed(page.decode("utf-8", errors="replace"))
    if not parser.candidates:
        raise ValueError("source page does not expose an og:image or twitter:image preview")
    return urljoin(final_page_url, parser.candidates[0])


def is_supported_image(payload: bytes, content_type: str) -> bool:
    jpeg = payload.startswith(b"\xff\xd8\xff") and payload.endswith(b"\xff\xd9")
    png = payload.startswith(b"\x89PNG\r\n\x1a\n") and payload.endswith(
        b"IEND\xaeB`\x82"
    )
    webp = (
        len(payload) >= 12
        and payload[:4] == b"RIFF"
        and payload[8:12] == b"WEBP"
        and int.from_bytes(payload[4:8], "little") + 8 == len(payload)
    )
    return content_type in {"image/jpeg", "image/png", "image/webp"} and (jpeg or png or webp)


def load_sources(manifest: Path, statuses: set[str]) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["review_status"] in statuses]
    if not selected:
        raise ValueError(f"manifest has no rows with status: {', '.join(sorted(statuses))}")
    ids = [row["id"] for row in selected]
    filenames = [row["filename"] for row in selected]
    if len(ids) != len(set(ids)) or len(filenames) != len(set(filenames)):
        raise ValueError("manifest IDs and filenames must be unique")
    return selected


def load_previous_asset_urls(report_path: Path) -> dict[str, str]:
    """Preserve resolved URLs when an existing local file is reused."""
    if not report_path.is_file():
        return {}
    try:
        records = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(record["id"]): str(record["assetUrl"])
        for record in records
        if isinstance(record, dict) and record.get("id") and record.get("assetUrl")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download license-documented Saudi-domain image candidates."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/saudi/sources.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/saudi/images/raw"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/saudi/download-report.json")
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["candidate", "approved"],
        help="Manifest review_status values to download.",
    )
    parser.add_argument("--limit", type=int, help="Download only the first N selected rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    try:
        rows = load_sources(args.manifest, set(args.statuses))
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Could not load manifest {args.manifest}: {error}") from error
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Selected {len(rows)} source(s) from {args.manifest}")
    if args.dry_run:
        for row in rows:
            print(f"{row['id']}: {row['source_page']} -> {args.output / row['filename']}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    previous_asset_urls = load_previous_asset_urls(args.report)
    report: list[dict[str, object]] = []
    failures = 0

    for index, row in enumerate(rows, start=1):
        destination = args.output / row["filename"]
        if destination.exists() and not args.overwrite:
            payload = destination.read_bytes()
            content_type = mimetypes.guess_type(destination.name)[0] or ""
            if is_supported_image(payload, content_type):
                existing_record: dict[str, object] = {
                    "id": row["id"],
                    "status": "existing",
                    "file": destination.as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "sourcePage": row["source_page"],
                }
                asset_url = row["asset_url"] or previous_asset_urls.get(row["id"])
                if asset_url:
                    existing_record["assetUrl"] = asset_url
                report.append(existing_record)
                print(f"[{index}/{len(rows)}] existing: {destination}")
                continue
            print(f"[{index}/{len(rows)}] invalid existing image; downloading again: {destination}")

        try:
            asset_url = resolve_asset_url(row["source_page"], row["asset_url"])
            payload, content_type, final_asset_url = request_bytes(asset_url)
            if not is_supported_image(payload, content_type):
                raise ValueError(f"unexpected image response: {content_type}")

            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            digest = hashlib.sha256(payload).hexdigest()
            report.append(
                {
                    "id": row["id"],
                    "status": "downloaded",
                    "file": destination.as_posix(),
                    "bytes": len(payload),
                    "sha256": digest,
                    "sourcePage": row["source_page"],
                    "assetUrl": final_asset_url,
                }
            )
            print(f"[{index}/{len(rows)}] downloaded: {destination} ({len(payload)} bytes)")
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            failures += 1
            report.append(
                {
                    "id": row["id"],
                    "status": "failed",
                    "sourcePage": row["source_page"],
                    "error": str(error),
                }
            )
            print(f"[{index}/{len(rows)}] failed {row['id']}: {error}")
        time.sleep(0.25)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report}")
    if failures:
        raise SystemExit(f"{failures} download(s) failed; see {args.report}")


if __name__ == "__main__":
    main()
