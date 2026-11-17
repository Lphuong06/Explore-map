"""ActiveSIGMA-inspired point evidence, combined with MAV map confidence.

Dirichlet evidence is attached to observed anchor samples, never to unseen sides.
MAV confidence is a relative reliability proxy, not a calibrated probability.
"""
import json
from collections import OrderedDict
import numpy as np

DEFAULTS = dict(
    alpha_prior=0.001,
    min_evidence=1.5,
    entropy_threshold=0.55,
    jsd_threshold=0.2,
    uncertain_fraction=0.25,
    conflict_fraction=0.2,
    geometry_min_valid_fraction=0.1,
    geometry_min_conf_ratio=0.75,
    geometry_score_threshold=0.5,
    max_anchor_samples=128,
)


def distribution(p):
    p = np.maximum(np.asarray(p, np.float64), 0)
    return p / max(float(p.sum()), 1e-12)


def entropy(p):
    p = distribution(p)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(max(2, len(p))))


def divergence(p, q):
    p, q = distribution(p), distribution(q)
    m = (p + q) / 2
    return float(
        (
            np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)))
            + np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)))
        )
        / (2 * np.log(2))
    )


def geometry_metrics(confidence, threshold, finite_mask, valid_mask):
    conf = np.asarray(confidence, np.float64)
    finite = np.asarray(finite_mask, bool) & np.isfinite(conf)
    valid = np.asarray(valid_mask, bool) & finite
    ratio = (
        float(np.median(conf[finite]) / max(float(threshold), 1e-12))
        if finite.any()
        else 0.0
    )
    return dict(
        source="MAV_MapAnything_relative_confidence",
        calibrated=False,
        finite_fraction=float(finite.mean()),
        valid_fraction=float(valid.mean()),
        median_confidence_ratio=ratio,
        raw_confidence_threshold=float(threshold),
    )


class RegionEvidence:
    def __init__(self, connection, config=None, *, explain=False):
        self.db = connection
        self.cfg = {**DEFAULTS, **(config or {})}
        self.explain = explain
        if set(self.cfg) - set(DEFAULTS):
            raise ValueError("Unknown region evidence configuration")
        for key, value in self.cfg.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid region setting {key}")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS region_samples (region TEXT, sample INTEGER, alpha BLOB, mass REAL, observations INTEGER, PRIMARY KEY(region,sample))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS region_decisions (frame TEXT,region TEXT,decision TEXT,PRIMARY KEY(frame,region))"
        )
        self.cache = OrderedDict()

    def _get(self, region, sample, classes):
        key = (region, int(sample))
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        row = self.db.execute(
            "SELECT alpha,mass,observations FROM region_samples WHERE region=? AND sample=?",
            key,
        ).fetchone()
        value = (
            (np.frombuffer(row[0], dtype="<f4").astype(float), row[1], row[2])
            if row
            else (np.full(classes, self.cfg["alpha_prior"]), 0.0, 0)
        )
        if len(value[0]) != classes:
            raise ValueError("Semantic class space changed inside a region")
        self.cache[key] = value
        while len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return value

    def _update(self, region, sample, probs, weight):
        alpha, mass, n = self._get(region, sample, len(probs))
        old = alpha / alpha.sum()
        conflict = divergence(probs, old) if n else None
        strength = max(0.0, float(weight)) * (1.0 - entropy(probs))
        alpha = alpha + strength * distribution(probs)
        mass += strength
        n += 1
        value = (alpha, mass, n)
        self.cache[(region, int(sample))] = value
        self.db.execute(
            "INSERT OR REPLACE INTO region_samples VALUES (?,?,?,?,?)",
            (region, int(sample), alpha.astype("<f4").tobytes(), mass, n),
        )
        result = dict(mass=mass, observations=n, entropy=entropy(alpha), jsd=conflict)
        if self.explain:

            def top(p):
                p = distribution(p)
                ids = np.argsort(p)[-3:][::-1]
                return [dict(class_id=int(i), probability=float(p[i])) for i in ids]

            result["explanation"] = dict(
                anchor_sample_index=int(sample),
                previous_top=top(old) if n > 1 else [],
                current_top=top(probs),
                posterior_top=top(alpha),
                current_entropy=entropy(probs),
                geometry_weight=float(weight),
                added_strength=float(strength),
                mass=float(mass),
                observations=int(n),
                posterior_entropy=result["entropy"],
                jsd=conflict,
            )
        return result

    def observe(self, record, decision, group_ids, anchors, old_records):
        cfg = self.cfg
        geometry = record.get("geometry")
        regions = []
        for gid in group_ids:
            anchor = anchors[gid]
            if anchor == record["frame"]:
                ids = np.unique(record.get("local_samples", []))[
                    : int(cfg["max_anchor_samples"])
                ]
                if len(ids) == 0 and len(record["points"]):
                    ids = np.linspace(
                        0, len(record["points"]) - 1, min(32, len(record["points"]))
                    ).astype(int)
                states = []
                for idx in ids:
                    state = self._update(
                        gid, int(idx), record["probs"][idx], self._weight(record, idx)
                    )
                    if self.explain:
                        state["explanation"]["current_sample_index"] = int(idx)
                    states.append(state)
                association = (
                    "first_observation" if not old_records else "unverified_association"
                )
                residual = None
            else:
                member = next(
                    (m for m in decision["scene_members"] if m["frame"] == anchor), None
                )
                if member is None:
                    continue
                old = old_records[anchor]
                states = []
                # Pairs are verified image/3D correspondences, not histogram similarity.
                for qi, ai in member["sample_pairs"]:
                    _, _, n = self._get(gid, ai, record["probs"].shape[1])
                    if n == 0:
                        self._update(gid, ai, old["probs"][ai], self._weight(old, ai))
                    state = self._update(
                        gid, ai, record["probs"][qi], self._weight(record, qi)
                    )
                    if self.explain:
                        state["explanation"]["current_sample_index"] = int(qi)
                    states.append(state)
                association = "verified_shared_surface"
                residual = member.get("world_residual_p90")
            region = self._decide(gid, states, geometry, association, residual)
            if self.explain:
                region["anchor_frame"] = anchor
                region["sample_diagnostics"] = [
                    state["explanation"] for state in states
                ]
                region["decision_thresholds"] = dict(cfg)
            self.db.execute(
                "INSERT OR REPLACE INTO region_decisions VALUES (?,?,?)",
                (record["frame"], gid, json.dumps(region)),
            )
            regions.append(region)
        return dict(
            decision,
            policy="semantic_geometry_regions_v6",
            region_decisions=regions,
            **self.combine_regions(regions),
        )

    @staticmethod
    def combine_regions(regions):
        def aggregate(values):
            return (
                True
                if any(v is True for v in values)
                else (None if not values or any(v is None for v in values) else False)
            )

        final = aggregate([r["need_more_observation"] for r in regions])
        result = dict(
            need_more_observation=final,
            status="observe_more"
            if final is True
            else ("sufficient" if final is False else "insufficient_evidence"),
            reasons=list(
                dict.fromkeys(reason for r in regions for reason in r["reasons"])
            ),
        )
        for component in ("semantic_decision", "geometry_decision"):
            parts = [r[component] for r in regions]
            result[component] = dict(
                need_more_observation=aggregate(
                    [p["need_more_observation"] for p in parts]
                ),
                reasons=list(
                    dict.fromkeys(reason for p in parts for reason in p["reasons"])
                ),
            )
            if component == "semantic_decision":
                result[component]["evaluable"] = bool(parts) and all(
                    p["evaluable"] for p in parts
                )
                result[component]["comparison_available"] = bool(parts) and all(
                    p["comparison_available"] for p in parts
                )
                result[component]["evidence_evaluable"] = bool(parts) and all(
                    p["evidence_evaluable"] for p in parts
                )
        return result

    @staticmethod
    def _weight(record, index):
        scores = record.get("geometry_weights")
        return float(scores[index]) if scores is not None else 0.0

    def _decide(self, gid, states, geometry, association, residual):
        c = self.cfg
        masses = [s["mass"] for s in states]
        entropies = [s["entropy"] for s in states]
        jsds = [s["jsd"] for s in states if s["jsd"] is not None]
        enough = (
            float(np.mean(np.asarray(masses) >= c["min_evidence"])) if masses else 0.0
        )
        uncertain = (
            float(np.mean(np.asarray(entropies) >= c["entropy_threshold"]))
            if entropies
            else None
        )
        conflict = (
            float(np.mean(np.asarray(jsds) >= c["jsd_threshold"])) if jsds else None
        )
        score = None
        if geometry is not None:
            score = min(
                1.0,
                geometry["valid_fraction"] / c["geometry_min_valid_fraction"],
                geometry["median_confidence_ratio"] / c["geometry_min_conf_ratio"],
            )
            if residual is not None:
                score = min(score, float(np.exp(-((residual / 0.15) ** 2))))
        # Evidence shortage independently requests more data; consistency still needs a shared surface.
        comparison_available = association == "verified_shared_surface" and bool(jsds)
        comparison_reliable = (
            comparison_available
            and score is not None
            and score >= c["geometry_score_threshold"]
        )
        semantic_value = None
        if (
            comparison_reliable
            and uncertain is not None
            and uncertain >= c["uncertain_fraction"]
        ):
            semantic_value = True
            semantic_reasons = ["semantic_uncertainty"]
        elif (
            comparison_reliable
            and conflict is not None
            and conflict >= c["conflict_fraction"]
        ):
            semantic_value = True
            semantic_reasons = ["semantic_inconsistency"]
        elif enough < 0.75:
            semantic_value = True
            semantic_reasons = ["insufficient_semantic_evidence"]
        elif association != "verified_shared_surface":
            semantic_reasons = ["no_verified_shared_surface"]
        elif not jsds:
            semantic_reasons = ["semantic_comparison_unavailable"]
        elif not comparison_reliable:
            semantic_reasons = ["semantic_comparison_geometry_unreliable"]
        else:
            semantic_value = False
            semantic_reasons = ["consistent_semantics_with_sufficient_evidence"]
        geometry_value = (
            None if score is None else bool(score < c["geometry_score_threshold"])
        )
        geometry_reasons = (
            ["geometry_confidence_unavailable"]
            if score is None
            else (
                ["geometric_unreliability"]
                if geometry_value
                else ["geometry_supported"]
            )
        )
        if semantic_value is True or geometry_value is True:
            value = True
            reasons = []
            if semantic_value is True:
                reasons.extend(semantic_reasons)
            if geometry_value is True:
                reasons.extend(geometry_reasons)
        elif semantic_value is None or geometry_value is None:
            value = None
            reasons = []
            if semantic_value is None:
                reasons.extend(semantic_reasons)
            if geometry_value is None:
                reasons.extend(geometry_reasons)
        else:
            value = False
            reasons = ["semantic_and_geometry_supported"]
        return dict(
            region_id=gid,
            association=association,
            need_more_observation=value,
            reasons=reasons,
            semantic_decision=dict(
                need_more_observation=semantic_value,
                evaluable=semantic_value is not None,
                comparison_available=comparison_available,
                comparison_reliable=bool(comparison_reliable),
                evidence_evaluable=True,
                reasons=semantic_reasons,
            ),
            geometry_decision=dict(
                need_more_observation=geometry_value, reasons=geometry_reasons
            ),
            semantic_consistency=dict(
                jsd_p90=float(np.percentile(jsds, 90)) if jsds else None,
                conflicting_fraction=conflict,
                comparable_samples=len(jsds),
            ),
            semantic_evidence=dict(
                model="confidence_weighted_Dirichlet",
                samples=len(states),
                median_strength=float(np.median(masses)) if masses else None,
                sufficient_fraction=enough,
                uncertain_fraction=uncertain,
            ),
            geometry_confidence=dict(
                metrics=geometry,
                score=score,
                calibrated=False,
                world_residual_p90=residual,
            ),
        )
