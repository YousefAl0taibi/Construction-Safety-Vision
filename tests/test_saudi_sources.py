import argparse
import base64
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

import scripts.download_saudi_sources as downloader
from scripts.download_saudi_sources import (
    SocialImageParser,
    is_supported_image,
    load_sources,
    request_bytes,
)
from scripts.run_saudi_pilot import load_manifest_entries


def test_social_image_parser_prefers_declared_candidates():
    parser = SocialImageParser()
    parser.feed(
        '<html><head><meta property="og:image" content="https://example.test/image.jpg">'
        '<meta name="twitter:image" content="https://example.test/other.jpg"></head></html>'
    )

    assert parser.candidates == [
        "https://example.test/image.jpg",
        "https://example.test/other.jpg",
    ]


def test_supported_image_requires_matching_type_and_magic():
    assert is_supported_image(b"\xff\xd8\xffpayload\xff\xd9", "image/jpeg")
    assert not is_supported_image(b"<html>error</html>", "image/jpeg")
    assert not is_supported_image(b"\xff\xd8\xfftruncated", "image/jpeg")
    assert not is_supported_image(b"\xff\xd8\xffpayload\xff\xd9", "text/html")


def test_downloader_retries_temporary_rate_limit(monkeypatch):
    class Headers:
        def get(self, key):
            return None

        def get_content_type(self):
            return "image/jpeg"

    class Response:
        headers = Headers()

        def __init__(self):
            self.reads = iter([b"\xff\xd8\xffpayload", b""])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            return next(self.reads)

        def geturl(self):
            return "https://example.test/image.jpg"

    attempts = 0

    def fake_urlopen(_request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPError(
                "https://example.test/image.jpg", 429, "rate limited", Headers(), None
            )
        return Response()

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)
    monkeypatch.setattr(downloader.time, "sleep", lambda _seconds: None)

    payload, content_type, final_url = request_bytes(
        "https://example.test/image.jpg", retries=1
    )

    assert attempts == 2
    assert payload == b"\xff\xd8\xffpayload"
    assert content_type == "image/jpeg"
    assert final_url == "https://example.test/image.jpg"


def test_saudi_manifest_has_unique_sources_and_expected_review_counts():
    manifest = Path("data/saudi/sources.csv")
    rows = load_sources(manifest, {"candidate", "approved", "excluded"})

    assert len(rows) == 21
    assert len({row["scene_group"] for row in rows}) == 5
    assert sum(row["review_status"] == "approved" for row in rows) == 15
    assert sum(row["review_status"] == "excluded" for row in rows) == 5
    assert sum(row["review_status"] == "candidate" for row in rows) == 1
    assert all(row["source_page"] and row["asset_url"] for row in rows)


def test_saudi_pilot_selects_only_approved_sources_by_default():
    rows = load_manifest_entries(Path("data/saudi/sources.csv"), {"approved"})

    assert len(rows) == 15
    assert {row["review_status"] for row in rows} == {"approved"}


def test_downloader_rejects_unknown_review_status():
    with pytest.raises(ValueError, match="no rows with status"):
        load_sources(Path("data/saudi/sources.csv"), {"typo"})


def test_existing_file_keeps_asset_url_in_custom_report(tmp_path, monkeypatch):
    manifest = tmp_path / "sources.csv"
    output = tmp_path / "raw"
    report = tmp_path / "reports" / "download.json"
    output.mkdir()
    manifest.write_text(
        "id,filename,scene_group,source_kind,source_page,asset_url,location,date,"
        "creator,rights,rights_url,review_status,notes\n"
        "sample,sample.png,scene,test,https://example.test/page,"
        "https://example.test/sample.png,Riyadh,2026-01-01,Creator,CC0,"
        "https://example.test/rights,approved,test image\n",
        encoding="utf-8",
    )
    output.joinpath("sample.png").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAusB9Y9Z4K0AAAAASUVORK5CYII="
        )
    )
    args = argparse.Namespace(
        manifest=manifest,
        output=output,
        report=report,
        statuses=["approved"],
        limit=None,
        overwrite=False,
        dry_run=False,
    )
    monkeypatch.setattr(downloader, "parse_args", lambda: args)

    downloader.main()

    records = json.loads(report.read_text(encoding="utf-8"))
    assert records[0]["status"] == "existing"
    assert records[0]["assetUrl"] == "https://example.test/sample.png"
