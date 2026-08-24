"""Internal assembler for canonical ``rtichoke_viz`` ReportSpec objects.

The assembler composes complete standalone canonical component specs. It does
not calculate statistics, normalize component specs, hoist evaluations, or
create report-global semantic registries.

This path is intentionally separate from the existing public summary-report
API, which currently delegates to the historical R backend and Quarto
composition. A future migration can replace that composition layer with the
shared browser ``renderReport()`` once an immutable vendored ``rtichoke_viz``
release exposes the report renderer. The existing public report behavior is
left unchanged here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

_SUPPORTED_COMPONENT_TYPES = {
    "performance_table",
    "roc",
    "calibration",
    "precision_recall",
    "gains",
    "lift",
}
_COMPONENT_ID_BASES = {
    "performance_table": "performance-table",
    "roc": "roc",
    "calibration": "calibration",
    "precision_recall": "precision-recall",
    "gains": "gains",
    "lift": "lift",
}


class _ReportComponentInput(TypedDict, total=False):
    spec: Mapping[str, object]
    title: str


class _ReportComponent(TypedDict, total=False):
    id: str
    title: str
    spec: Mapping[str, object]


class _ReportSpec(TypedDict, total=False):
    schemaVersion: str
    type: str
    title: str
    components: list[_ReportComponent]


def _report_spec_from_components(
    components: Sequence[_ReportComponentInput],
    *,
    title: str | None = None,
) -> _ReportSpec:
    """Compose complete canonical component specs into a ReportSpec.

    Component order is preserved exactly. Component IDs are deterministic and
    live in a report-local identity domain: the first component of a type gets
    its boring base ID (for example ``roc``), and repeats get ``-2``, ``-3``,
    and so on. Embedded specs are retained as-is, so evaluation IDs remain
    component-local even when equal strings occur in multiple components.
    """
    if not components:
        raise ValueError("ReportSpec requires at least one component")

    type_counts: dict[str, int] = {}
    report_components: list[_ReportComponent] = []
    for component in components:
        spec = component.get("spec")
        if spec is None:
            raise ValueError("Report component is missing spec")

        component_type = spec.get("type")
        if not isinstance(component_type, str):
            raise ValueError("Report component spec is missing a string type")
        if component_type not in _SUPPORTED_COMPONENT_TYPES:
            raise ValueError(f"Unsupported ReportSpec component type: {component_type}")

        count = type_counts.get(component_type, 0) + 1
        type_counts[component_type] = count
        base_id = _COMPONENT_ID_BASES[component_type]
        component_id = base_id if count == 1 else f"{base_id}-{count}"

        assembled: _ReportComponent = {
            "id": component_id,
            "spec": spec,
        }
        component_title = component.get("title")
        if component_title is not None:
            assembled["title"] = component_title
        report_components.append(assembled)

    report: _ReportSpec = {
        "schemaVersion": "1.0",
        "type": "report",
        "components": report_components,
    }
    if title is not None:
        report["title"] = title
    return report
