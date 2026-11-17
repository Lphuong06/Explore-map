"""SIGMA-inspired external semantic voxel evidence for MAV.

Association uses current world coordinates, not image retrieval or frame anchors.
Direction bins are 12 Fibonacci-sphere directions (not HEALPix). Only observed
bins contribute disagreement: unseen directions never trigger an observation.
Camera-frame samples remain authoritative; pose corrections subtract/reintegrate
per-frame contributions. Historical decisions retain arrival-time meaning.
"""
from pathlib import Path
import hashlib, json, sqlite3, time
import numpy as np
from ma_slam.region_evidence import geometry_metrics

POLICY = "semantic_world_voxel_v1"
DEFAULTS = dict(
    voxel_size=0.10,
    direction_bins=12,
    direction_neighbors=4,
    max_range=10.0,
    alpha_prior=0.001,
    min_evidence=1.5,
    entropy_threshold=0.55,
    jsd_threshold=0.2,
    uncertain_fraction=0.25,
    conflict_fraction=0.2,
    required_sufficient_fraction=0.75,
    geometry_min_valid_fraction=0.1,
    geometry_min_conf_ratio=0.75,
    geometry_score_threshold=0.5,
    diagnostic_samples=128,
)


def normalize(p):
    p = np.maximum(np.asarray(p, np.float64), 0)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)


def entropy(p):
    p = normalize(p)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(max(2, p.shape[-1]))


def jsd(p, q):
    p, q = normalize(p), normalize(q)
    m = (p + q) / 2
    return (
        (p * np.log(np.maximum(p, 1e-12) / np.maximum(m, 1e-12))).sum(-1)
        + (q * np.log(np.maximum(q, 1e-12) / np.maximum(m, 1e-12))).sum(-1)
    ) / (2 * np.log(2))


def tri(values):
    return (
        True
        if any(v is True for v in values)
        else None
        if not values or any(v is None for v in values)
        else False
    )


class VoxelObservationDatabase:
    def __init__(self, root, config=None, *, resume=False, frame_selection="all"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = {**DEFAULTS, **(config or {})}
        if set(self.cfg) - set(DEFAULTS):
            raise ValueError("Unknown voxel evidence configuration")
        if frame_selection not in ("all", "first"):
            raise ValueError("Invalid frame selection")
        for key, value in self.cfg.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid setting " + key)
        for key in ("direction_bins", "direction_neighbors", "diagnostic_samples"):
            if int(self.cfg[key]) != self.cfg[key]:
                raise ValueError("Integer setting required: " + key)
        self.frame_selection = frame_selection
        signature = dict(
            policy=POLICY, frame_selection=frame_selection, config=self.cfg
        )
        file = self.root / "voxel_config.json"
        if resume and (not file.exists() or json.loads(file.read_text()) != signature):
            raise ValueError("Cannot resume with different voxel policy/configuration")
        self.db = sqlite3.connect(self.root / "observations.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(
            """CREATE TABLE IF NOT EXISTS frames(frame TEXT PRIMARY KEY,submap INTEGER,graph_key INTEGER,artifact TEXT,decision TEXT,pose TEXT);
CREATE TABLE IF NOT EXISTS memberships(submap INTEGER,frame TEXT,graph_key INTEGER,PRIMARY KEY(submap,frame));
CREATE TABLE IF NOT EXISTS region_decisions(frame TEXT,region TEXT,decision TEXT,PRIMARY KEY(frame,region));
CREATE TABLE IF NOT EXISTS voxel_states(x INTEGER,y INTEGER,z INTEGER,evidence BLOB,mass REAL,observations INTEGER,PRIMARY KEY(x,y,z)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS direction_states(x INTEGER,y INTEGER,z INTEGER,bin INTEGER,evidence BLOB,mass REAL,observations INTEGER,PRIMARY KEY(x,y,z,bin)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS frame_voxels(frame_id INTEGER,x INTEGER,y INTEGER,z INTEGER,bin INTEGER,weight REAL,PRIMARY KEY(frame_id,x,y,z)) WITHOUT ROWID;"""
        )
        if not resume and self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]:
            self.db.close()
            raise ValueError("Use a new output directory for the voxel run")
        file.write_text(json.dumps(signature, indent=2))
        (self.root / "selection.json").write_text(
            json.dumps(dict(frame_selection=frame_selection))
        )
        n = int(self.cfg["direction_bins"])
        k = np.arange(n)
        z = 1 - 2 * (k + 0.5) / n
        theta = np.pi * (3 - np.sqrt(5)) * k
        r = np.sqrt(1 - z * z)
        self.axes = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
        dots = self.axes @ self.axes.T
        np.fill_diagonal(dots, -np.inf)
        self.neighbors = np.argsort(-dots, axis=1)[
            :, : min(int(self.cfg["direction_neighbors"]), n - 1)
        ]
        self.voxels = {}
        self.directions = {}
        self.dirty_voxels = set()
        self.dirty_directions = set()
        self.records = []
        self.record_by_frame = {}
        self.classes = None
        self.timings = dict(
            pose_reindex_seconds=0.0,
            voxel_assessment_seconds=0.0,
            artifact_io_seconds=0.0,
            pose_changed_frames=0,
            assignment_changed_frames=0,
        )
        if resume:
            for frame, key, artifact, decision, pose in self.db.execute(
                "SELECT frame,graph_key,artifact,decision,pose FROM frames ORDER BY rowid"
            ):
                row = json.loads(decision)
                with np.load(self.root / artifact) as data:
                    record = dict(
                        frame=frame,
                        key=key,
                        frame_id=int(row["frame_id"]),
                        points=data["camera_points"].copy(),
                        probs=data["probabilities"].copy(),
                        weights=data["geometry_weights"].copy(),
                        pixel_indices=data["pixel_indices"].copy(),
                        pose=np.asarray(json.loads(pose)),
                        geometry=json.loads(str(data["geometry_metrics"])),
                    )
                self._check_classes(record)
                record["contribution"] = self._contribution(record, record["pose"])
                self._apply(record, 1)
                self.records.append(record)
                self.record_by_frame[frame] = record
            self._flush_states()
            self.db.commit()

    def _check_classes(self, record):
        classes = record["probs"].shape[1]
        if self.classes is not None and self.classes != classes:
            raise ValueError("Semantic class space changed")
        self.classes = classes

    def _assign(self, record, pose):
        world = record["points"] @ pose[:3, :3].T + pose[:3, 3]
        keys = np.floor(world / self.cfg["voxel_size"]).astype(np.int64)
        centers = (keys + 0.5) * self.cfg["voxel_size"]
        vectors = centers - pose[:3, 3]
        bins = np.argmax(vectors @ self.axes.T, axis=1).astype(np.int64)
        signature = hashlib.sha256(keys.tobytes() + bins.tobytes()).digest()
        return keys, bins, signature

    def _contribution(self, record, pose, assigned=None):
        keys, bins, signature = (
            self._assign(record, pose) if assigned is None else assigned
        )
        if not len(keys):
            return dict(
                keys=np.empty((0, 3), np.int64),
                bins=np.empty(0, np.int64),
                probs=np.empty((0, self.classes)),
                weights=np.empty(0),
                representatives=np.empty(0, np.int64),
                signature=signature,
            )
        unique, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        order = np.argsort(inverse, kind="stable")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        p = normalize(record["probs"])
        omega = np.clip(record["weights"], 0, 1) * (1 - entropy(p))
        sums = np.add.reduceat(omega[order], starts)
        q = np.add.reduceat((p * omega[:, None])[order], starts, axis=0)
        empty = sums <= 1e-12
        if empty.any():
            q[empty] = np.add.reduceat(p[order], starts, axis=0)[empty]
        q = normalize(q)
        representatives = order[starts]
        return dict(
            keys=unique,
            bins=bins[representatives],
            probs=q,
            weights=sums / counts,
            representatives=representatives,
            signature=signature,
        )

    def _state(self, table, key):
        if key not in table:
            table[key] = dict(
                evidence=np.zeros(self.classes, np.float64), mass=0.0, frames=set()
            )
        return table[key]

    def _apply(self, record, sign):
        c = record["contribution"]
        fid = record["frame_id"]
        for key, bin, p, w in zip(c["keys"], c["bins"], c["probs"], c["weights"]):
            key = tuple(map(int, key))
            dk = (*key, int(bin))
            for table, k, dirty in (
                (self.voxels, key, self.dirty_voxels),
                (self.directions, dk, self.dirty_directions),
            ):
                state = self._state(table, k)
                state["evidence"] += sign * w * p
                state["mass"] += sign * float(w)
                if sign > 0:
                    state["frames"].add(fid)
                else:
                    state["frames"].discard(fid)
                if not state["frames"]:
                    del table[k]
                else:
                    state["evidence"] = np.maximum(state["evidence"], 0)
                    state["mass"] = max(0.0, state["mass"])
                dirty.add(k)

    def _mean(self, state):
        return normalize(state["evidence"] + self.cfg["alpha_prior"])

    def _write_membership(self, record):
        c = record["contribution"]
        fid = record["frame_id"]
        self.db.execute("DELETE FROM frame_voxels WHERE frame_id=?", (fid,))
        self.db.executemany(
            "INSERT INTO frame_voxels VALUES (?,?,?,?,?,?)",
            [
                (fid, int(k[0]), int(k[1]), int(k[2]), int(b), float(w))
                for k, b, w in zip(c["keys"], c["bins"], c["weights"])
            ],
        )

    def _flush_states(self):
        for table, dirty, name in (
            (self.voxels, self.dirty_voxels, "voxel_states"),
            (self.directions, self.dirty_directions, "direction_states"),
        ):
            updates = []
            deletes = []
            for k in dirty:
                state = table.get(k)
                if state is None:
                    deletes.append(k)
                else:
                    updates.append(
                        (
                            *k,
                            state["evidence"].astype("<f8").tobytes(),
                            state["mass"],
                            len(state["frames"]),
                        )
                    )
            size = 3 if name == "voxel_states" else 4
            self.db.executemany(
                f"INSERT OR REPLACE INTO {name} VALUES ({','.join(['?']*(size+3))})",
                updates,
            )
            conditions = " AND ".join(
                f"{k}=?"
                for k in (["x", "y", "z"] if size == 3 else ["x", "y", "z", "bin"])
            )
            self.db.executemany(f"DELETE FROM {name} WHERE {conditions}", deletes)
            dirty.clear()

    def refresh_poses(self, graph):
        start = time.perf_counter()
        updates = []
        for record in self.records:
            pose = np.asarray(graph.get_pose(record["key"]), np.float64)
            if np.allclose(pose, record["pose"], rtol=0, atol=1e-9):
                continue
            self.timings["pose_changed_frames"] += 1
            assigned = self._assign(record, pose)
            if assigned[2] != record["contribution"]["signature"]:
                self._apply(record, -1)
                record["contribution"] = self._contribution(record, pose, assigned)
                self._apply(record, 1)
                self._write_membership(record)
                self.timings["assignment_changed_frames"] += 1
            record["pose"] = pose.copy()
            updates.append((json.dumps(pose.tolist()), record["frame"]))
        self.db.executemany("UPDATE frames SET pose=? WHERE frame=?", updates)
        self._flush_states()
        self.db.commit()
        self.timings["pose_reindex_seconds"] += time.perf_counter() - start

    def _observe(self, record):
        c = record["contribution"]
        cfg = self.cfg
        states = []
        for key, bin, p, w, rep in zip(
            c["keys"], c["bins"], c["probs"], c["weights"], c["representatives"]
        ):
            key = tuple(map(int, key))
            bin = int(bin)
            old = self.voxels.get(key)
            previous = self._mean(old) if old is not None else None
            temporal = float(jsd(p, previous)) if previous is not None else None
            old_dir = self.directions.get((*key, bin))
            dir_evidence = (
                old_dir["evidence"] if old_dir is not None else np.zeros(self.classes)
            ) + w * p
            dir_mean = normalize(dir_evidence + cfg["alpha_prior"])
            neighbor_means = []
            for nb in self.neighbors[bin]:
                neighbor = self.directions.get((*key, int(nb)))
                if neighbor is not None and neighbor["mass"] > 0:
                    neighbor_means.append(self._mean(neighbor))
            directional = (
                float(np.mean([jsd(dir_mean, q) for q in neighbor_means]))
                if neighbor_means
                else None
            )
            e = (old["evidence"] if old is not None else np.zeros(self.classes)) + w * p
            mass = (old["mass"] if old is not None else 0.0) + float(w)
            posterior = normalize(e + cfg["alpha_prior"])
            available = [x for x in [temporal, directional] if x is not None]
            pixel = int(record["pixel_indices"][rep])
            prior_frames = len(old["frames"]) if old is not None else 0
            states.append(
                dict(
                    voxel=list(key),
                    direction_bin=bin,
                    current_sample_index=int(rep),
                    pixel_index=pixel,
                    mass=mass,
                    prior_observations=prior_frames,
                    observations=prior_frames + 1,
                    entropy=float(entropy(posterior)),
                    temporal_jsd=temporal,
                    directional_jsd=directional,
                    jsd=max(available) if available else None,
                    directional_heat=0.0
                    if directional is None
                    else directional / (float(dir_evidence.sum()) + 1.0),
                    previous_top=self._top(previous) if previous is not None else [],
                    current_top=self._top(p),
                    posterior_top=self._top(posterior),
                    previous_frame_id=max(old["frames"]) if old is not None else None,
                    added_strength=float(w),
                )
            )
        self._apply(record, 1)
        return self._decide(record, states), states

    def _top(self, p):
        ids = np.argsort(p)[-3:][::-1]
        return [dict(class_id=int(i), probability=float(p[i])) for i in ids]

    def _decide(self, record, states):
        cfg = self.cfg
        n = len(states)
        comparable = [s for s in states if s["jsd"] is not None]
        enough = (
            float(np.mean([s["mass"] >= cfg["min_evidence"] for s in states]))
            if n
            else 0.0
        )
        uncertain = (
            float(np.mean([s["entropy"] >= cfg["entropy_threshold"] for s in states]))
            if n
            else None
        )
        conflict = (
            float(np.mean([s["jsd"] >= cfg["jsd_threshold"] for s in comparable]))
            if comparable
            else None
        )
        m = record["geometry"]
        score = (
            min(
                1.0,
                m["valid_fraction"] / cfg["geometry_min_valid_fraction"],
                m["median_confidence_ratio"] / cfg["geometry_min_conf_ratio"],
            )
            if m is not None
            else None
        )
        reliable = score is not None and score >= cfg["geometry_score_threshold"]
        semantic = None
        if (
            reliable
            and comparable
            and uncertain is not None
            and uncertain >= cfg["uncertain_fraction"]
        ):
            semantic = True
            sr = ["semantic_uncertainty"]
        elif (
            reliable
            and comparable
            and conflict is not None
            and conflict >= cfg["conflict_fraction"]
        ):
            semantic = True
            sr = ["semantic_inconsistency"]
        elif enough < cfg["required_sufficient_fraction"]:
            semantic = True
            sr = ["insufficient_semantic_evidence"]
        elif not comparable:
            sr = ["no_previous_world_voxel_evidence"]
        elif not reliable:
            sr = ["semantic_comparison_geometry_unreliable"]
        else:
            semantic = False
            sr = ["consistent_semantics_with_sufficient_evidence"]
        geom = None if score is None else bool(score < cfg["geometry_score_threshold"])
        gr = (
            ["geometry_confidence_unavailable"]
            if geom is None
            else ["geometric_unreliability"]
            if geom
            else ["geometry_supported"]
        )
        if semantic is True or geom is True:
            final = True
            reasons = (sr if semantic is True else []) + (gr if geom is True else [])
        elif semantic is None or geom is None:
            final = None
            reasons = (sr if semantic is None else []) + (gr if geom is None else [])
        else:
            final = False
            reasons = ["semantic_and_geometry_supported"]
        new = sum(s["prior_observations"] == 0 for s in states)
        diagnostic = sorted(
            states, key=lambda s: (s["jsd"] or 0, s["entropy"]), reverse=True
        )[: int(cfg["diagnostic_samples"])]
        region = dict(
            region_id="observed_world_voxels",
            association="world_voxel_lookup",
            need_more_observation=final,
            reasons=reasons,
            semantic_decision=dict(
                need_more_observation=semantic,
                reasons=sr,
                evaluable=semantic is not None,
                comparison_available=bool(comparable),
                evidence_evaluable=True,
                comparison_reliable=bool(reliable and comparable),
            ),
            geometry_decision=dict(need_more_observation=geom, reasons=gr),
            semantic_consistency=dict(
                jsd_p90=float(np.percentile([s["jsd"] for s in comparable], 90))
                if comparable
                else None,
                conflicting_fraction=conflict,
                comparable_samples=len(comparable),
                association_source="pose_depth_world_voxel",
                independently_verified_correspondences=False,
                directional_comparable_voxels=sum(
                    s["directional_jsd"] is not None for s in states
                ),
                missing_direction_penalty=False,
            ),
            semantic_evidence=dict(
                model="per_frame_per_world_voxel_weighted_Dirichlet",
                samples=n,
                median_strength=float(np.median([s["mass"] for s in states]))
                if n
                else None,
                sufficient_fraction=enough,
                uncertain_fraction=uncertain,
                new_voxels=new,
                reused_voxels=n - new,
            ),
            geometry_confidence=dict(
                metrics=m, score=score, calibrated=False, world_residual_p90=None
            ),
            sample_diagnostics=diagnostic,
            decision_thresholds=dict(cfg),
            directional_heat_p90=float(
                np.percentile([s["directional_heat"] for s in states], 90)
            )
            if n
            else 0.0,
        )
        return dict(
            policy=POLICY,
            association="world_voxel_lookup",
            frame_selection=self.frame_selection,
            need_more_observation=final,
            reasons=reasons,
            status="observe_more"
            if final is True
            else "sufficient"
            if final is False
            else "unknown",
            semantic_decision=region["semantic_decision"],
            geometry_decision=region["geometry_decision"],
            region_decisions=[region],
            observed_voxels=n,
            reused_voxels=n - new,
            new_voxels=new,
            scene_group_ids=[],
            pose_basis="current_graph_before_inserting_this_frame",
            image_retrieval_for_semantics=False,
            frame_anchor_required=False,
            direction_binning="Fibonacci_sphere_nearest_axis",
            directional_heat_is_diagnostic=True,
        )

    def add_submap(self, sm, graph, semantic_processor):
        self.refresh_poses(graph)
        results = []
        selected = (
            range(min(1, len(sm.image_paths)))
            if self.frame_selection == "first"
            else range(len(sm.image_paths))
        )
        for i in selected:
            frame = str(Path(sm.image_paths[i]).resolve())
            self.db.execute(
                "INSERT OR IGNORE INTO memberships VALUES (?,?,?)",
                (int(sm.base_id), frame, int(sm.key(i))),
            )
            if frame in self.record_by_frame:
                continue
            source = semantic_processor.observation_paths[str(sm.image_paths[i])]
            with np.load(source) as data:
                indices = data["pixel_indices"]
                probs = normalize(data["probabilities"]).astype(np.float32)
                labels = data["class_ids"]
            if labels.shape != sm.conf[i].shape:
                raise ValueError("Semantic/geometry grids differ")
            local = sm.points[i].reshape(-1, 3)[indices]
            inverse = np.linalg.inv(sm.poses_local[i])
            camera = local @ inverse[:3, :3].T + inverse[:3, 3]
            raw = sm.conf[i].reshape(-1)[indices]
            finite = np.isfinite(camera).all(-1) & (camera[:, 2] > 0)
            valid = (
                finite
                & (np.linalg.norm(camera, axis=1) <= self.cfg["max_range"])
                & (raw > sm.conf_threshold * 0.75)
                & np.isfinite(probs).all(-1)
                & (probs.sum(-1) > 0)
                & (labels.reshape(-1)[indices] != 0)
            )
            if sm.mask is not None:
                valid &= sm.mask[i].reshape(-1)[indices].astype(bool)
            record = dict(
                frame=frame,
                frame_id=int(sm.frame_ids[i]),
                key=int(sm.key(i)),
                points=camera[valid],
                probs=probs[valid],
                weights=np.clip(
                    raw[valid] / max(float(sm.conf_threshold), 1e-12), 0, 1
                ),
                pixel_indices=indices[valid],
                pose=np.asarray(graph.get_pose(sm.key(i))).copy(),
                geometry=geometry_metrics(raw, sm.conf_threshold, finite, valid),
            )
            self._check_classes(record)
            start = time.perf_counter()
            record["contribution"] = self._contribution(record, record["pose"])
            decision, states = self._observe(record)
            self.timings["voxel_assessment_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            dest = self.root / f"submap_{sm.base_id}" / f"frame_{i}.npz"
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                dest,
                camera_points=record["points"],
                probabilities=record["probs"],
                geometry_weights=record["weights"],
                pixel_indices=record["pixel_indices"],
                class_ids=labels,
                source=str(source),
                geometry_metrics=json.dumps(record["geometry"]),
                voxel_keys=record["contribution"]["keys"],
                direction_bins=record["contribution"]["bins"],
                mass_after=np.array([s["mass"] for s in states]),
                jsd_before=np.array(
                    [np.nan if s["jsd"] is None else s["jsd"] for s in states]
                ),
                posterior_entropy=np.array([s["entropy"] for s in states]),
                prior_observations=np.array([s["prior_observations"] for s in states]),
            )
            decision.update(
                frame=frame,
                frame_id=record["frame_id"],
                submap_id=int(sm.base_id),
                graph_key=record["key"],
                submap_frame_index=int(i),
                artifact=str(dest.relative_to(self.root)),
            )
            self.db.execute(
                "INSERT INTO frames VALUES (?,?,?,?,?,?)",
                (
                    frame,
                    int(sm.base_id),
                    record["key"],
                    str(dest.relative_to(self.root)),
                    json.dumps(decision),
                    json.dumps(record["pose"].tolist()),
                ),
            )
            self.db.execute(
                "INSERT INTO region_decisions VALUES (?,?,?)",
                (
                    frame,
                    "observed_world_voxels",
                    json.dumps(decision["region_decisions"][0]),
                ),
            )
            self._write_membership(record)
            self.records.append(record)
            self.record_by_frame[frame] = record
            self.db.commit()
            with (self.root / "decisions.jsonl").open("a") as f:
                f.write(json.dumps(decision) + "\n")
            self.timings["artifact_io_seconds"] += time.perf_counter() - start
            results.append(decision)
            print(
                f"[voxel_reobserve] frame={record['frame_id']} voxels={decision['observed_voxels']} reused={decision['reused_voxels']} need_more={decision['need_more_observation']} reasons={','.join(decision['reasons'])}",
                flush=True,
            )
        self._flush_states()
        self.db.commit()
        return results

    def close(self):
        self._flush_states()
        self.db.commit()
        self.db.close()
