"""Auditable train-only exclusions based on release and source metadata."""

import re
from collections import Counter

FAMILY_KEYS = {"exclude_s1mb_japanese_source_family", "exclude_jdmb_source_family"}

LIST_KEYS = {"exclude_subsets", "exclude_dataset_groups", "exclude_source_patterns"}
IDENTITY_KEYS = {
    "dataset",
    "original_dataset",
    "repo",
    "name",
    "source_url",
    "url",
    "original_url",
    "original_subset",
    "subset",
    "original_file",
    "source_family",
    "source_id",
}


def validate_filter(policy):
    if not isinstance(policy, dict) or set(policy) - LIST_KEYS - FAMILY_KEYS:
        raise ValueError("Invalid dataset.train_filter keys")
    for key in FAMILY_KEYS:
        if type(policy.get(key, False)) is not bool:
            raise ValueError(key + " must be a boolean")
    if FAMILY_KEYS <= policy.keys() and len({policy[key] for key in FAMILY_KEYS}) > 1:
        raise ValueError("Conflicting source-family filter aliases")
    for key in LIST_KEYS:
        values = policy.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(key + " must be a list of nonempty strings")
    for pattern in policy.get("exclude_source_patterns", []):
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError("Invalid source pattern: " + pattern) from exc


def source_identities(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"teacher", "teacher_models", "generation", "prompt_revision"}:
                continue
            if key in IDENTITY_KEYS and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from source_identities(item)
    elif isinstance(value, list):
        for item in value:
            yield from source_identities(item)


def filter_training_records(rows, policy):
    """Preserve row/target/order; remove matches and connected parent/group rows."""
    validate_filter(policy)
    patterns = [re.compile(p, re.IGNORECASE) for p in policy.get("exclude_source_patterns", [])]
    direct = set()
    reasons = {}
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("parent_id", "group_id"):
            if row.get(key) is not None:
                identity = (key, row[key])
                if identity in seen:
                    parents[root(i)] = root(seen[identity])
                else:
                    seen[identity] = i
        matches = []
        if any(policy.get(key) for key in FAMILY_KEYS):
            if type(row.get("jdmb_source_family")) is not bool:
                raise ValueError("Missing/invalid jdmb_source_family: " + row["id"])
            if row["jdmb_source_family"]:
                matches.append("jdmb_source_family")
        if row.get("subset") in policy.get("exclude_subsets", []):
            matches.append("subset")
        if row.get("dataset_group") in policy.get("exclude_dataset_groups", []):
            matches.append("dataset_group")
        evidence = "\n".join([row.get("subset", ""), *source_identities(row.get("provenance", {}))])
        matches.extend("source_pattern:" + p.pattern for p in patterns if p.search(evidence))
        if matches:
            direct.add(i)
            reasons[row["id"]] = matches
    removed_roots = {root(i) for i in direct}
    excluded = [r for i, r in enumerate(rows) if root(i) in removed_roots]
    kept = [r for i, r in enumerate(rows) if root(i) not in removed_roots]
    if not kept:
        raise ValueError("No training records remain after filtering")
    receipt = {
        "policy": policy,
        "before": len(rows),
        "remaining": len(kept),
        "excluded": len(excluded),
        "direct_matches": len(direct),
        "excluded_ids": [r["id"] for r in excluded],
        "direct_match_reasons": reasons,
        "excluded_subsets": dict(sorted(Counter(r.get("subset", "") for r in excluded).items())),
        "remaining_types": dict(Counter(r["input"]["type"] for r in kept)),
        "remaining_marked_source_family_rows": sum(
            r.get("jdmb_source_family") is True for r in kept
        ),
        "scope": "Recorded source identities and connected train groups; not semantic overlap or pretraining.",
    }
    return kept, receipt
