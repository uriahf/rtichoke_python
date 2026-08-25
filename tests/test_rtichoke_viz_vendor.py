import hashlib
import tarfile
from pathlib import Path


_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"
_RELEASE_DIR = "rtichoke-viz-0.7.0"
_SHA256 = "f09c30e231a8be39c2e89ba6ae39c90ed8cab67021213e17681a475066a9806e"
_SOURCE_COMMIT = "b3564d2824ec1791f791fda406c99b3d7865a68f"


def test_vendored_rtichoke_viz_v070_provenance_archive_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "release=v0.7.0" in provenance
    assert f"source_commit={_SOURCE_COMMIT}" in provenance
    assert "archive=rtichoke-viz-0.7.0.tar.gz" in provenance
    assert f"sha256={_SHA256}" in provenance

    archive = _VENDOR / "rtichoke-viz-0.7.0.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == _SHA256
    with tarfile.open(archive, "r:gz") as release:
        assert set(release.getnames()) == {
            _RELEASE_DIR,
            f"{_RELEASE_DIR}/MANIFEST",
            f"{_RELEASE_DIR}/rtichoke-viz.css",
            f"{_RELEASE_DIR}/rtichoke-viz.js",
            f"{_RELEASE_DIR}/rtichoke-viz.schema.json",
            f"{_RELEASE_DIR}/rtichoke-viz-v2.schema.json",
        }
        manifest = release.extractfile(f"{_RELEASE_DIR}/MANIFEST")
        assert manifest is not None
        assert manifest.read().decode() == (f"version=0.7.0\ncommit={_SOURCE_COMMIT}\n")
        for filename in (
            "rtichoke-viz.css",
            "rtichoke-viz.js",
            "rtichoke-viz.schema.json",
            "rtichoke-viz-v2.schema.json",
        ):
            packaged = release.extractfile(f"{_RELEASE_DIR}/{filename}")
            assert packaged is not None
            assert (_VENDOR / filename).read_bytes() == packaged.read()

    assert not (_VENDOR / "rtichoke-viz-0.6.0.tar.gz").exists()
    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema
    assert '"decision_curve"' in v2_schema
    assert '"interventions_avoided"' in v2_schema


def test_v070_bundle_keeps_existing_exports_and_adds_interventions_avoided():
    bundle = (_VENDOR / "rtichoke-viz.js").read_text(encoding="utf-8")
    for export_name in (
        "renderRoc",
        "renderCalibration",
        "RtichokeChartSpecSchema",
        "renderGainsV2",
        "renderLiftV2",
        "renderDecisionCurveV2",
        "DecisionCurveV2SpecSchema",
        "renderInterventionsAvoidedV2",
        "InterventionsAvoidedV2SpecSchema",
        "RtichokeChartSpecV2Schema",
        "renderPerformanceTable",
        "renderReport",
    ):
        assert export_name in bundle

    assert "Fixed Time Horizon" in bundle
