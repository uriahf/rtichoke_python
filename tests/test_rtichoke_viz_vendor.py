import hashlib
import tarfile
from pathlib import Path


_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"


def test_vendored_rtichoke_viz_v031_provenance_archive_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "release=v0.3.1" in provenance
    assert "source_commit=5ccde928a0bf9fa6ece2b7572687b442c57a98a9" in provenance
    assert "archive=rtichoke-viz-0.3.1.tar.gz" in provenance
    assert (
        "sha256=121aa8eb8d0f8427ecfb2c01dab0fb05668eaedf47ddcfc0cd282a7ecf1ce448"
        in provenance
    )

    archive = _VENDOR / "rtichoke-viz-0.3.1.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "121aa8eb8d0f8427ecfb2c01dab0fb05668eaedf47ddcfc0cd282a7ecf1ce448"
    )
    with tarfile.open(archive, "r:gz") as release:
        assert set(release.getnames()) == {
            "rtichoke-viz-0.3.1",
            "rtichoke-viz-0.3.1/MANIFEST",
            "rtichoke-viz-0.3.1/rtichoke-viz.css",
            "rtichoke-viz-0.3.1/rtichoke-viz.js",
            "rtichoke-viz-0.3.1/rtichoke-viz.schema.json",
            "rtichoke-viz-0.3.1/rtichoke-viz-v2.schema.json",
        }

    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema


def test_v031_bundle_keeps_v1_adds_v2_exports_and_time_horizon_control():
    bundle = (_VENDOR / "rtichoke-viz.js").read_text(encoding="utf-8")
    for export_name in (
        "renderRoc",
        "renderCalibration",
        "RtichokeChartSpecSchema",
        "renderGainsV2",
        "RtichokeChartSpecV2Schema",
    ):
        assert export_name in bundle

    assert "Fixed Time Horizon" in bundle
