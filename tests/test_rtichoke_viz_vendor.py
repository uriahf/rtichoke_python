import hashlib
import tarfile
from pathlib import Path


_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"
_RELEASE_DIR = "rtichoke-viz-0.4.0"


def test_vendored_rtichoke_viz_v040_provenance_archive_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "release=v0.4.0" in provenance
    assert "source_commit=2125c7099839cadd536c6f38f3f7e23a17ca4348" in provenance
    assert "archive=rtichoke-viz-0.4.0.tar.gz" in provenance
    assert (
        "sha256=9a687cd938f1875d577e592664ca75447455166169f6132dd7f79406515e14e1"
        in provenance
    )

    archive = _VENDOR / "rtichoke-viz-0.4.0.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "9a687cd938f1875d577e592664ca75447455166169f6132dd7f79406515e14e1"
    )
    with tarfile.open(archive, "r:gz") as release:
        assert set(release.getnames()) == {
            _RELEASE_DIR,
            f"{_RELEASE_DIR}/MANIFEST",
            f"{_RELEASE_DIR}/rtichoke-viz.css",
            f"{_RELEASE_DIR}/rtichoke-viz.js",
            f"{_RELEASE_DIR}/rtichoke-viz.schema.json",
            f"{_RELEASE_DIR}/rtichoke-viz-v2.schema.json",
        }
        for filename in (
            "rtichoke-viz.css",
            "rtichoke-viz.js",
            "rtichoke-viz.schema.json",
            "rtichoke-viz-v2.schema.json",
        ):
            packaged = release.extractfile(f"{_RELEASE_DIR}/{filename}")
            assert packaged is not None
            assert (_VENDOR / filename).read_bytes() == packaged.read()

    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema


def test_v040_bundle_keeps_existing_exports_adds_lift_and_time_horizon_control():
    bundle = (_VENDOR / "rtichoke-viz.js").read_text(encoding="utf-8")
    for export_name in (
        "renderRoc",
        "renderCalibration",
        "RtichokeChartSpecSchema",
        "renderGainsV2",
        "renderLiftV2",
        "RtichokeChartSpecV2Schema",
    ):
        assert export_name in bundle

    assert "Fixed Time Horizon" in bundle
