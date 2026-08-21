from pathlib import Path


_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"


def test_vendored_rtichoke_viz_v020_provenance_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "release=v0.2.0" in provenance
    assert "source_commit=45dc109a6a0679d0f8f3f9452d8de9306a89b906" in provenance
    assert "archive=rtichoke-viz-0.2.0.tar.gz" in provenance
    assert "sha256=3861277c01b3983f8b344a9ee0237c7d09fd0ba4c3d1e0cce489962e7b559d9f" in provenance

    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema


def test_v020_bundle_keeps_v1_browser_exports():
    bundle = (_VENDOR / "rtichoke-viz.js").read_text()
    for export_name in ("renderRoc", "renderCalibration", "RtichokeChartSpecSchema"):
        assert export_name in bundle
