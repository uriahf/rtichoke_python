import hashlib
import tarfile
from pathlib import Path

_VENDOR = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"
_RELEASE_DIR = "rtichoke-viz-0.24.2"
_SHA256 = "de0d658406ea7d7725b02d24a58ff306992b7d30afd592d49c5d223b2793aee8"
_SOURCE_COMMIT = "bb5dc5ed4f4d32e85df4df3d737661a74a3a862d"


def test_vendored_rtichoke_viz_v0242_provenance_archive_and_schemas():
    provenance = (_VENDOR / "VENDORED_FROM").read_text()
    assert "version=0.24.2" in provenance
    assert "tag=v0.24.2" in provenance
    assert "tag_commit=26e603cf2c3664ce063b4f020e17aab4243addf6" in provenance
    assert "snapshot=" not in provenance
    assert f"source_commit={_SOURCE_COMMIT}" in provenance
    assert "archive=rtichoke-viz-0.24.2.tar.gz" in provenance
    assert f"sha256={_SHA256}" in provenance

    assert (_VENDOR / "MANIFEST").read_text() == (
        f"version=0.24.2\ncommit={_SOURCE_COMMIT}\n"
    )
    assert (_VENDOR / "rtichoke-viz-0.24.2.tar.gz.sha256").read_text().split()[
        0
    ] == _SHA256
    archive = _VENDOR / "rtichoke-viz-0.24.2.tar.gz"
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
            f"version=0.24.2\ncommit={_SOURCE_COMMIT}\n"
        )
        for filename in (
            "MANIFEST",
            "rtichoke-viz.css",
            "rtichoke-viz.js",
            "rtichoke-viz.schema.json",
            "rtichoke-viz-v2.schema.json",
            "rtichoke-viz-report.schema.json",
        ):
            packaged = release.extractfile(f"{_RELEASE_DIR}/{filename}")
            assert packaged is not None
            assert (_VENDOR / filename).read_bytes() == packaged.read()

    assert not (_VENDOR / "rtichoke-viz-0.24.1.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.24.0.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.23.2.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.23.0.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.22.3.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.22.2.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.22.1.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.22.0.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.20.2.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.20.1.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.20.0.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.19.0.tar.gz").exists()
    assert not (_VENDOR / "rtichoke-viz-0.14.0.tar.gz").exists()
    assert (_VENDOR / "rtichoke-viz.js").stat().st_size > 0
    assert (_VENDOR / "rtichoke-viz.css").stat().st_size > 0

    v1_schema = (_VENDOR / "rtichoke-viz.schema.json").read_text()
    v2_schema = (_VENDOR / "rtichoke-viz-v2.schema.json").read_text()
    report_schema = (_VENDOR / "rtichoke-viz-report.schema.json").read_text()
    assert '"$id": "https://rtichoke.dev/schema/viz/1.0.json"' in v1_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/2.0.json"' in v2_schema
    assert '"$id": "https://rtichoke.dev/schema/viz/report.json"' in report_schema
    for schema in (v2_schema, report_schema):
        assert '"estimateStatus"' in schema
        for status in (
            "defined",
            "undefined_empty_population",
            "beyond_support_plateau",
        ):
            assert f'"{status}"' in schema
    assert '"decision_curve"' in v2_schema
    assert '"interventions_avoided"' in v2_schema
    assert '"summary_metrics"' in report_schema


def test_v0242_bundle_keeps_existing_exports_and_outcome_distribution():
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
        "renderPredictionDistribution",
        "PredictionDistributionSpecSchema",
        "renderOutcomeDistribution",
        "renderContinuousOutcomeDistribution",
        "OutcomeDistributionSpecSchema",
    ):
        assert export_name in bundle

    assert "Fixed Time Horizon" in bundle
    assert "Event Probability" in bundle
    assert "Prediction Percentile" in bundle
    assert "Prediction Score" in bundle
