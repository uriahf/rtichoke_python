import hashlib
import tarfile
from pathlib import Path


_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"


def test_vendored_rtichoke_viz_v030_provenance_archive_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "release=v0.3.0" in provenance
    assert "source_commit=aca9188ea856167557efb20980a0b43e0481b8c8" in provenance
    assert "archive=rtichoke-viz-0.3.0.tar.gz" in provenance
    assert (
        "sha256=558f8d9e16f9544659b84e33f72511065163291a1b97a3c5511b61d1e1f0cac1"
        in provenance
    )

    archive = _VENDOR / "rtichoke-viz-0.3.0.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "558f8d9e16f9544659b84e33f72511065163291a1b97a3c5511b61d1e1f0cac1"
    )
    with tarfile.open(archive, "r:gz") as release:
        assert set(release.getnames()) == {
            "rtichoke-viz-0.3.0",
            "rtichoke-viz-0.3.0/MANIFEST",
            "rtichoke-viz-0.3.0/rtichoke-viz.css",
            "rtichoke-viz-0.3.0/rtichoke-viz.js",
            "rtichoke-viz-0.3.0/rtichoke-viz.schema.json",
            "rtichoke-viz-0.3.0/rtichoke-viz-v2.schema.json",
        }

    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema


def test_v030_bundle_keeps_v1_and_adds_v2_browser_exports():
    bundle = (_VENDOR / "rtichoke-viz.js").read_text(encoding="utf-8")
    for export_name in (
        "renderRoc",
        "renderCalibration",
        "RtichokeChartSpecSchema",
        "renderGainsV2",
        "RtichokeChartSpecV2Schema",
    ):
        assert export_name in bundle
