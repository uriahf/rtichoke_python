"""Internal assembler for canonical ``rtichoke_viz`` ReportSpec v1.1 objects.

The assembler composes complete standalone canonical component specs into a structured
ReportSpec v1.1 object hierarchy with sections, items (components and groups), and title.
It does not calculate statistics, normalize component specs, hoist evaluations, or
create report-global semantic registries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

_V10_SCHEMA_TYPES = {
    "summary_metrics",
}

_V20_SCHEMA_TYPES = {
    "performance_table",
    "roc",
    "calibration",
    "precision_recall",
    "gains",
    "lift",
    "decision_curve",
    "interventions_avoided",
}

_ALL_SUPPORTED_TYPES = _V10_SCHEMA_TYPES | _V20_SCHEMA_TYPES


def _validate_spec_schema_version(spec: Mapping[str, object]) -> None:
    """Validate that spec schemaVersion strictly matches type requirements.

    summary_metrics -> "1.0" or "1.1"
    all v2 component types -> "2.0"
    """
    spec_type = spec.get("type")
    if not isinstance(spec_type, str):
        raise ValueError("Report component spec is missing a string type")
    if spec_type not in _ALL_SUPPORTED_TYPES:
        raise ValueError(f"Unsupported ReportSpec component type: {spec_type}")

    schema_version = spec.get("schemaVersion")
    if not isinstance(schema_version, str):
        raise ValueError("Report component spec is missing a string schemaVersion")

    if spec_type in _V10_SCHEMA_TYPES:
        if schema_version not in {"1.0", "1.1"}:
            raise ValueError(
                f"Component type {spec_type!r} requires schemaVersion '1.0' or '1.1', got {schema_version!r}"
            )
    elif spec_type in _V20_SCHEMA_TYPES:
        if schema_version != "2.0":
            raise ValueError(
                f"Component type {spec_type!r} requires schemaVersion '2.0', got {schema_version!r}"
            )


class _ReportComponentV11(TypedDict, total=False):
    type: str
    id: str
    title: str
    spec: Mapping[str, object]


class _ReportGroupV11(TypedDict, total=False):
    type: str
    id: str
    title: str
    components: list[_ReportComponentV11]


class _ReportSectionV11(TypedDict, total=False):
    id: str
    title: str
    items: list[_ReportComponentV11 | _ReportGroupV11]


class _ReportSpecV11(TypedDict, total=False):
    schemaVersion: str
    type: str
    title: str
    sections: list[_ReportSectionV11]


def _build_report_spec_v11(
    sections: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
) -> _ReportSpecV11:
    """Compose structured ReportSpec v1.1 hierarchy.

    Validates type-aware schemaVersion for every embedded component spec.
    """
    if not sections:
        raise ValueError("ReportSpec v1.1 requires at least one section")

    assembled_sections: list[_ReportSectionV11] = []

    for section in sections:
        sec_id = section.get("id")
        if not isinstance(sec_id, str) or not sec_id:
            raise ValueError("Report section must have a non-empty string id")

        sec_title = section.get("title")
        if sec_title is None:
            sec_title = sec_id
        elif not isinstance(sec_title, str) or not sec_title:
            raise ValueError("Report section must have a non-empty string title")

        items_assembled: list[_ReportComponentV11 | _ReportGroupV11] = []

        components_raw = section.get("components")
        if components_raw is not None:
            if not isinstance(components_raw, Sequence):
                raise ValueError("Section components must be a sequence")
            for comp in components_raw:
                if not isinstance(comp, Mapping):
                    raise ValueError("Report component must be a Mapping")
                comp_id = comp.get("id")
                raw_spec = comp.get("spec")
                if not isinstance(comp_id, str) or not comp_id:
                    raise ValueError("Report component must have a non-empty string id")
                if not isinstance(raw_spec, Mapping):
                    raise ValueError("Report component is missing spec")
                _validate_spec_schema_version(cast(Mapping[str, object], raw_spec))
                assembled_comp: _ReportComponentV11 = {
                    "type": "component",
                    "id": comp_id,
                    "spec": raw_spec,
                }
                comp_title = comp.get("title")
                if comp_title is not None:
                    if not isinstance(comp_title, str):
                        raise ValueError("Report component title must be a string")
                    assembled_comp["title"] = comp_title
                items_assembled.append(assembled_comp)

        groups_raw = section.get("groups")
        if groups_raw is not None:
            if not isinstance(groups_raw, Sequence):
                raise ValueError("Section groups must be a sequence")
            for group in groups_raw:
                if not isinstance(group, Mapping):
                    raise ValueError("Report group must be a Mapping")
                group_id = group.get("id")
                group_title = group.get("title")
                if not isinstance(group_id, str) or not group_id:
                    raise ValueError("Report group must have a non-empty string id")
                if not isinstance(group_title, str) or not group_title:
                    raise ValueError("Report group must have a non-empty string title")

                grp_comps_raw = group.get("components")
                if not isinstance(grp_comps_raw, Sequence) or not grp_comps_raw:
                    raise ValueError("Group components must be a non-empty sequence")

                grp_components: list[_ReportComponentV11] = []
                for comp in grp_comps_raw:
                    if not isinstance(comp, Mapping):
                        raise ValueError("Report component must be a Mapping")
                    comp_id = comp.get("id")
                    raw_spec = comp.get("spec")
                    if not isinstance(comp_id, str) or not comp_id:
                        raise ValueError(
                            "Report component must have a non-empty string id"
                        )
                    if not isinstance(raw_spec, Mapping):
                        raise ValueError("Report component is missing spec")
                    _validate_spec_schema_version(cast(Mapping[str, object], raw_spec))
                    assembled_comp = cast(
                        _ReportComponentV11,
                        {
                            "type": "component",
                            "id": comp_id,
                            "spec": raw_spec,
                        },
                    )
                    comp_title = comp.get("title")
                    if comp_title is not None:
                        if not isinstance(comp_title, str):
                            raise ValueError("Report component title must be a string")
                        assembled_comp["title"] = comp_title
                    grp_components.append(assembled_comp)

                grp_dict: _ReportGroupV11 = {
                    "type": "group",
                    "id": group_id,
                    "title": group_title,
                    "components": grp_components,
                }
                items_assembled.append(grp_dict)

        if not items_assembled:
            raise ValueError(f"Section {sec_id!r} has no components or groups")

        assembled_sec: _ReportSectionV11 = {
            "id": sec_id,
            "title": sec_title,
            "items": items_assembled,
        }
        assembled_sections.append(assembled_sec)

    report: _ReportSpecV11 = {
        "schemaVersion": "1.1",
        "type": "report",
        "sections": assembled_sections,
    }
    if title is not None:
        report["title"] = title
    return report
