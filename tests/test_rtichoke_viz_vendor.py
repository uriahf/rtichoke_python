import hashlib
import tarfile
from pathlib import Path

_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"
_RELEASE_DIR = "rtichoke-viz-0.14.0"
_SHA256 = "b7c30d12db4b3f8008ef035bd4f7e04c09d599ced5bc8606a4a4141c5cdf48ee"
_SOURCE_COMMIT = "ec3a382656ce5b1c735175d5a8fd1ceb4f153eaf"


def test_vendored_rtichoke_viz_v0140_provenance_archive_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "release=v0.14.0" in provenance
    assert f"source_commit={_SOURCE_COMMIT}" in provenance
    assert "archive=rtichoke-viz-0.14.0.tar.gz" in provenance
    assert f"sha256={_SHA256}" in provenance

    archive = _VENDOR / "rtichoke-viz-0.14.0.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == _SHA256
    with tarfile.open(archive, "r:gz") as release:
        assert set(release.getnames()) == {
            _RELEASE_DIR,
            f"{_RELEASE_DIR}/MANIFEST",
            f"{_RELEASE_DIR}/rtichoke-viz.css",
            f"{_RELEASE_DIR}/rtichoke-viz.js",
            f"{_RELEASE_DIR}/rtichoke-viz.schema.json",
            f"{_RELEASE_DIR}/rtichoke-viz-v2.schema.json",
            f"{_RELEASE_DIR}/rtichoke-viz-report.schema.json",
        }
        manifest = release.extractfile(f"{_RELEASE_DIR}/MANIFEST")
        assert manifest is not None
        assert manifest.read().decode() == (
            f"version=0.14.0\ncommit={_SOURCE_COMMIT}\n"
        )
        for filename in (
            "rtichoke-viz.css",
            "rtichoke-viz.js",
            "rtichoke-viz.schema.json",
            "rtichoke-viz-v2.schema.json",
            "rtichoke-viz-report.schema.json",
        ):
            packaged = release.extractfile(f"{_RELEASE_DIR}/{filename}")
            assert packaged is not None
            assert (_VENDOR / filename).read_bytes() == packaged.read()

    assert not (_VENDOR / "rtichoke-viz-0.10.0.tar.gz").exists()
    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    report_schema = (_VENDOR / "rtichoke-viz-report.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/report.json"' in report_schema
    assert '"decision_curve"' in v2_schema
    assert '"interventions_avoided"' in v2_schema
    assert '"summary_metrics"' in report_schema


def test_v0140_bundle_keeps_existing_exports_and_time_dependent_surfaces():
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
        "renderPrecisionRecallV2",
        "PrecisionRecallV2SpecSchema",
        "RtichokeChartSpecV2Schema",
        "renderPerformanceTable",
        "renderReport",
        "SummaryMetricsSpecSchema",
    ):
        assert export_name in bundle

    assert "Fixed Time Horizon" in bundle
