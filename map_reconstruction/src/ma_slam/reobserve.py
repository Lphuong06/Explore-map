"""Frame database and causal semantic re-observation advice; no semantic map fusion.

All-history visual retrieval is verified by local features and current MAV geometry.
Persistent overlapping groups require direct anchor evidence, never transitive union.
Decisions remain heuristics on observed shared surfaces.
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
from scipy.spatial import cKDTree
from ma_slam.visual_scene import VisualSceneIndex, local_features, verified_local_pairs


DEFAULTS = dict(match_distance=0.15, min_matches=30, min_overlap=0.2,
                max_camera_distance=8.0, max_neighbors=20, max_range=10.0,
                entropy_threshold=0.55, jsd_threshold=0.2,
                min_baseline=0.2, min_view_angle=15.0, min_independent_views=2)
DEFAULTS.update(retrieval_topk=50, retrieval_anchor_topk=10, retrieval_min_similarity=0.3,
                local_max_features=2000, local_max_pixel_distance=6.0, local_fast_threshold=10,
                local_ratio=0.8, min_local_matches=12, min_local_inlier_ratio=0.3)


def normalize(p):
    p = np.maximum(np.asarray(p, dtype=np.float32), 0)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)


def entropy(p):
    p = normalize(p)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(-1)


def jsd(p, q):
    p, q = normalize(p), normalize(q)
    return np.maximum(0, entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2) / np.log(2)


def semantic_advice(metrics, config=None):
    """Request another observation only for semantic evidence, never missing views.

    Insufficient evidence does not certify correctness and does not itself request
    a revisit. View diversity remains diagnostic, including for occluded objects.
    """
    cfg = {**DEFAULTS, **(config or {})}
    result = dict(metrics)
    result.update(policy='semantic_evidence_v2', need_more_observation=None)
    if result['valid_samples'] < cfg['min_matches']:
        result.update(status='insufficient_evidence', reasons=['insufficient_valid_geometry'])
        return result
    reasons = []
    if result['entropy'] is not None and result['entropy'] >= cfg['entropy_threshold']:
        reasons.append('semantic_uncertainty')
    if result['max_pair_jsd'] is not None and result['max_pair_jsd'] >= cfg['jsd_threshold']:
        reasons.append('cross_view_semantic_conflict')
    if reasons:
        result.update(status='observe_more', need_more_observation=True, reasons=reasons)
    elif not result['scene_members']:
        result.update(status='insufficient_evidence', reasons=['new_scene_no_verified_overlap'])
    else:
        result.update(status='sufficient', need_more_observation=False, reasons=['consistent_observed_semantics'])
    return result


class ObservationDatabase:
    def __init__(self, root, config=None, descriptor_provider=None, region_config=None, resume=False, frame_selection='all', explain=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if frame_selection not in ('all','first'):raise ValueError('frame_selection must be all or first')
        self.frame_selection=frame_selection
        selection_file=self.root/'selection.json'
        previous_selection=json.loads(selection_file.read_text())['frame_selection'] if selection_file.exists() else 'all'
        if resume and previous_selection!=frame_selection:
            raise ValueError('Cannot change frame selection when resuming an existing database')
        self.cfg = {**DEFAULTS, **(config or {})}
        unknown = set(self.cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown re-observation settings: {sorted(unknown)}')
        for key, value in self.cfg.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('min_matches', 'max_neighbors', 'min_independent_views', 'retrieval_topk',
                    'retrieval_anchor_topk', 'local_max_features', 'min_local_matches', 'local_fast_threshold'):
            if int(self.cfg[key]) != self.cfg[key]:
                raise ValueError(f'{key} must be an integer')
        for key in ('min_overlap', 'entropy_threshold', 'jsd_threshold',
                    'retrieval_min_similarity', 'local_ratio', 'min_local_inlier_ratio'):
            if self.cfg[key] > 1:
                raise ValueError(f'{key} must not exceed 1')
        self.db = sqlite3.connect(self.root / 'observations.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS frames (frame TEXT PRIMARY KEY, submap INTEGER, graph_key INTEGER, artifact TEXT, decision TEXT, pose TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS memberships (submap INTEGER, frame TEXT, graph_key INTEGER, PRIMARY KEY (submap, frame))')
        self.db.execute('CREATE TABLE IF NOT EXISTS scene_groups (group_id TEXT PRIMARY KEY, anchor_frame TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS scene_group_members (group_id TEXT, frame TEXT, PRIMARY KEY (group_id, frame))')
        self.db.execute('CREATE TABLE IF NOT EXISTS scene_edges (query_frame TEXT, match_frame TEXT, evidence TEXT, PRIMARY KEY (query_frame, match_frame))')
        if self.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0] and not resume:
            self.db.close()
            raise ValueError('Re-observation output already contains frames; use a new output directory')
        self.records = []
        self.visual_index = VisualSceneIndex()
        self.descriptor_provider = descriptor_provider
        self.groups = {}
        self.region_store = None
        if region_config and region_config.get('enable', True):
            from ma_slam.region_evidence import RegionEvidence
            self.region_store = RegionEvidence(self.db, {k:v for k,v in region_config.items() if k!='enable'},explain=explain)
        self.record_by_frame = {}
        if resume:
            self.groups = dict(self.db.execute('SELECT group_id,anchor_frame FROM scene_groups'))
            anchor_frames = set(self.groups.values())
            for frame,key,artifact in self.db.execute('SELECT frame,graph_key,artifact FROM frames ORDER BY rowid'):
                with np.load(self.root/artifact) as saved:
                    record = dict(frame=frame,key=key,points=saved['camera_points'],probs=normalize(saved['probabilities']),
                                  descriptor=saved['descriptor'],local_descriptors=saved['local_descriptors'],local_samples=saved['local_samples'])
                    if 'geometry_weights' in saved:
                        record['geometry_weights'] = saved['geometry_weights']
                        record['geometry'] = json.loads(str(saved['geometry_metrics']))
                self.records.append(record);self.record_by_frame[frame]=record
                self.visual_index.add(record['descriptor'],anchor=frame in anchor_frames)
        (self.root / 'config.json').write_text(json.dumps(self.cfg, indent=2))
        selection_file.write_text(json.dumps(dict(frame_selection=frame_selection),indent=2))

    def close(self):
        self.db.close()

    @staticmethod
    def world(record, graph):
        pose = np.asarray(graph.get_pose(record['key']))
        return record['points'] @ pose[:3, :3].T + pose[:3, 3], pose

    def refresh_poses(self, graph):
        # Every comparison reads current graph poses, including after loop closure.
        for record in self.records:
            pose = np.asarray(graph.get_pose(record['key']))
            self.db.execute('UPDATE frames SET pose=? WHERE frame=?',
                            (json.dumps(pose.tolist()), record['frame']))
        self.db.commit()

    def add_submap(self, sm, graph, semantic_processor):
        self.refresh_poses(graph)
        selected=list(range(min(1,len(sm.image_paths)))) if self.frame_selection=='first' else list(range(len(sm.image_paths)))
        if not selected:return []
        descriptors = getattr(sm, 'retrieval_vectors', None)
        if descriptors is None:
            if self.descriptor_provider is None:
                from ma_slam.loop import ImageRetrieval
                self.descriptor_provider = ImageRetrieval(device=semantic_processor.device)
            descriptors = self.descriptor_provider.describe([sm.image_paths[i] for i in selected])
            descriptors_by_frame=dict(zip(selected,np.asarray(descriptors,np.float32)))
        else:
            if np.asarray(descriptors).ndim!=2 or len(descriptors) not in (len(sm.image_paths),len(selected)):
                raise ValueError('Visual descriptors do not align with submap frames')
            descriptors_by_frame={i:descriptors[i] for i in selected}
        descriptors = np.asarray(descriptors, np.float32)
        if descriptors.ndim != 2 or len(descriptors) not in (len(sm.image_paths),len(selected)):
            raise ValueError('Visual descriptors do not align with submap frames')
        decisions = []
        for i in selected:
            path=sm.image_paths[i]
            frame = str(Path(path).resolve())
            self.db.execute('INSERT OR IGNORE INTO memberships VALUES (?,?,?)',
                            (int(sm.base_id), frame, int(sm.key(i))))
            self.db.commit()
            if frame in self.record_by_frame:
                continue  # Overlap is one physical observation, not new evidence.
            source = semantic_processor.observation_paths[str(path)]
            with np.load(source) as data:
                indices = data['pixel_indices']
                probs = normalize(data['probabilities'])
                labels = data['class_ids']
            if labels.shape != sm.conf[i].shape:
                raise ValueError('Observation grid differs from MAV geometry')
            local = sm.points[i].reshape(-1, 3)[indices]
            inverse = np.linalg.inv(sm.poses_local[i])
            camera = local @ inverse[:3, :3].T + inverse[:3, 3]
            valid = np.isfinite(camera).all(-1) & (camera[:, 2] > 0)
            valid &= np.linalg.norm(camera, axis=1) <= self.cfg['max_range']
            valid &= sm.conf[i].reshape(-1)[indices] > sm.conf_threshold * 0.75
            if sm.mask is not None:
                valid &= sm.mask[i].reshape(-1)[indices].astype(bool)
            valid &= np.isfinite(probs).all(-1) & (probs.sum(-1) > 0)
            valid &= labels.reshape(-1)[indices] != 0
            record = dict(frame=frame, key=sm.key(i), points=camera[valid], probs=probs[valid],
                          descriptor=VisualSceneIndex.vector(descriptors_by_frame[i])[0])
            from ma_slam.region_evidence import geometry_metrics
            raw_conf = sm.conf[i].reshape(-1)[indices]
            finite_geometry = np.isfinite(camera).all(-1) & (camera[:,2]>0)
            record['geometry'] = geometry_metrics(raw_conf,sm.conf_threshold,finite_geometry,valid)
            record['geometry_weights'] = np.clip(raw_conf[valid]/max(float(sm.conf_threshold),1e-12),0,1)
            record.update(local_features(sm.colors[i], indices[valid], labels.shape,
                                         self.cfg['local_max_features'], self.cfg['local_max_pixel_distance'],
                                         self.cfg['local_fast_threshold']))
            decision = self.evaluate(record, graph)
            group_ids = [gid for gid, anchor in self.groups.items()
                         if any(member['frame'] == anchor for member in decision['scene_members'])]
            is_anchor = not group_ids
            if is_anchor:
                gid = f'group_{sm.base_id}_{i}'
                self.groups[gid] = frame
                group_ids = [gid]
                self.db.execute('INSERT INTO scene_groups VALUES (?,?)', (gid, frame))
                # Each member shares directly verified evidence with this anchor.
                for member in decision['scene_members']:
                    self.db.execute('INSERT OR IGNORE INTO scene_group_members VALUES (?,?)', (gid, member['frame']))
            for gid in group_ids:
                self.db.execute('INSERT OR IGNORE INTO scene_group_members VALUES (?,?)', (gid, frame))
            for member in decision['scene_members']:
                self.db.execute('INSERT INTO scene_edges VALUES (?,?,?)', (frame, member['frame'], json.dumps(member)))
            if self.region_store is not None:
                decision = self.region_store.observe(record,decision,group_ids,self.groups,self.record_by_frame)
            dest = self.root / f'submap_{sm.base_id}' / f'frame_{i}.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, camera_points=record['points'], probabilities=record['probs'].astype(np.float16),
                                pixel_indices=indices[valid], class_ids=labels, source=str(source),
                                intrinsics=sm.intrinsics[i], descriptor=record['descriptor'],
                                local_descriptors=record['local_descriptors'], local_samples=record['local_samples'],
                                geometry_weights=record['geometry_weights'],geometry_metrics=json.dumps(record['geometry']))
            decision.update(frame=frame, frame_id=float(sm.frame_ids[i]), submap_id=int(sm.base_id),
                            graph_key=int(sm.key(i)), artifact=str(dest.relative_to(self.root)),
                            frame_selection=self.frame_selection,submap_frame_index=int(i))
            decision['scene_group_id'] = f'query_{sm.base_id}_{i}'
            decision['scene_group_ids'] = group_ids
            pose = np.asarray(graph.get_pose(sm.key(i)))
            self.db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?)',
                            (frame, int(sm.base_id), int(sm.key(i)), str(dest.relative_to(self.root)),
                             json.dumps(decision), json.dumps(pose.tolist())))
            self.db.commit()
            with (self.root / 'decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(decision) + '\n')
            self.records.append(record)
            self.record_by_frame[frame] = record
            self.visual_index.add(record['descriptor'], anchor=is_anchor)
            decisions.append(decision)
            print(f"[reobserve] frame={sm.frame_ids[i]} need_more={decision['need_more_observation']} "
                  f"status={decision['status']} reasons={','.join(decision['reasons'])}")
        return decisions

    def evaluate(self, record, graph):
        cfg = self.cfg
        points, pose = self.world(record, graph)
        result = dict(need_more_observation=False, status='insufficient_evidence', reasons=[],
                      valid_samples=len(points), scene_members=[], independent_views=1,
                      entropy=None, max_pair_jsd=None,
                      policy='semantic_evidence_v2', pose_basis=getattr(graph, 'pose_basis', 'current_optimized_graph_at_arrival'),
                      grouping='global_visual_local_3d_v3', retrieval_candidates=[])
        if len(points) < cfg['min_matches']:
            return semantic_advice(result, cfg)
        ent = float(np.mean(entropy(record['probs'])) / np.log(record['probs'].shape[-1]))
        result['entropy'] = ent
        # Keep direct callers/tests consistent with frames inserted via add_submap.
        for old in self.records[self.visual_index.count:]:
            self.visual_index.add(old['descriptor'], anchor=old['frame'] in self.groups.values())
        candidates = self.visual_index.query(record['descriptor'], cfg['retrieval_topk'],
                                              cfg['retrieval_anchor_topk'], cfg['retrieval_min_similarity'])
        independent = [pose[:3, 3]]
        center = np.mean(points, axis=0)
        pair_scores, matched_entropy = [], []
        for candidate in candidates:
            old = self.records[candidate['index']]
            trace = dict(frame=old['frame'], similarity=candidate['similarity'], sources=candidate['sources'])
            result['retrieval_candidates'].append(trace)
            old_points, old_pose = self.world(old, graph)
            if len(old_points) < cfg['min_matches']:
                trace['verification'] = 'insufficient_geometry'
                continue
            distances, matches = cKDTree(old_points).query(points)
            keep = distances <= cfg['match_distance']
            query_ids = np.flatnonzero(keep)
            # One old sample cannot inflate overlap by matching many query samples.
            _, unique = np.unique(matches[keep], return_index=True)
            query_ids = query_ids[unique]
            overlap = len(query_ids) / min(len(points), len(old_points))
            if len(query_ids) < cfg['min_matches'] or overlap < cfg['min_overlap']:
                trace['verification'] = 'no_shared_surface'
                continue
            local_pairs, local_total, local_ratio = verified_local_pairs(record, old, points, old_points, cfg)
            if len(local_pairs) < cfg['min_local_matches'] or local_ratio < cfg['min_local_inlier_ratio']:
                trace.update(verification='local_visual_geometry_rejected', local_matches=local_total,
                             local_inliers=len(local_pairs), local_inlier_ratio=local_ratio)
                continue
            trace['verification'] = 'accepted'
            previous = old['probs'][local_pairs[:, 1]]
            current = record['probs'][local_pairs[:, 0]]
            score = float(np.mean(jsd(current, previous)))
            residuals = np.linalg.norm(points[local_pairs[:,0]]-old_points[local_pairs[:,1]],axis=1)
            pair_scores.append(score)
            matched_entropy.append(float(np.mean(entropy(previous)) / np.log(previous.shape[-1])))
            position = old_pose[:3, 3]
            def distinct(other):
                a, b = center - position, center - other
                angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12), -1, 1)))
                return np.linalg.norm(position-other) >= cfg['min_baseline'] and angle >= cfg['min_view_angle']
            if all(distinct(other) for other in independent):
                independent.append(position)
            result['scene_members'].append(dict(frame=old['frame'], graph_key=int(old['key']),
                                                matched_samples=len(local_pairs), surface_matched_samples=len(query_ids),
                                                local_matches=local_total, local_inlier_ratio=local_ratio,
                                                retrieval_similarity=candidate['similarity'],
                                                sample_pairs=local_pairs.tolist(),world_residual_p90=float(np.percentile(residuals,90)),
                                                overlap=float(overlap), jsd=score))
        result['independent_views'] = len(independent)
        result['max_pair_jsd'] = max(pair_scores) if pair_scores else None
        # High uncertainty in any matched observation remains a reason to inspect.
        result['entropy'] = max([ent] + matched_entropy)
        return semantic_advice(result, cfg)
