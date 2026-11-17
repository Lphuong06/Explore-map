"""Causal scene grouping, ambiguity, overlap deduplication and pose correction."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ma_slam.reobserve import ObservationDatabase, jsd
from ma_slam.semantic import OneFormerSemantic
from ma_slam.visual_scene import VisualSceneIndex, verified_local_pairs


class Graph:
    def __init__(self):
        self.poses = {0: np.eye(4), 1: np.eye(4)}
        self.poses[1][0, 3] = 1
    def get_pose(self, key):
        return self.poses[key]


class ReobserveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = ObservationDatabase(self.tmp.name, dict(min_matches=5))
        self.graph = Graph()
        x, y = np.meshgrid(np.linspace(-.5, .5, 8), np.linspace(-.5, .5, 8))
        self.world = np.column_stack([x.ravel(), y.ravel(), np.full(64, 2.)])
    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()
    def record(self, key, probs=(.01, .98, .01)):
        return dict(frame=f'frame{key}', key=key,
                    points=self.world-self.graph.get_pose(key)[:3, 3],
                    probs=np.tile(probs, (64, 1)), descriptor=np.array([1., 0.]),
                    local_descriptors=np.random.default_rng(0).integers(0,256,(64,32),dtype=np.uint8),
                    local_samples=np.arange(64))
    def test_first_frame_is_unknown_without_requesting_revisit(self):
        out = self.db.evaluate(self.record(0), self.graph)
        self.assertFalse(out['need_more_observation'])
        self.assertEqual(out['status'], 'insufficient_evidence')
    def test_consistent_diverse_views_sufficient(self):
        self.db.records.append(self.record(0))
        out = self.db.evaluate(self.record(1), self.graph)
        self.assertFalse(out['need_more_observation'])
        self.assertEqual(out['independent_views'], 2)
    def test_same_scene_conflicting_labels_requests_more(self):
        self.db.records.append(self.record(0))
        out = self.db.evaluate(self.record(1, (.01, .01, .98)), self.graph)
        self.assertIn('cross_view_semantic_conflict', out['reasons'])
    def test_entropy_and_repeated_view_do_not_pass(self):
        self.graph.poses[1] = np.eye(4)
        self.db.records.append(self.record(0, (1/3, 1/3, 1/3)))
        out = self.db.evaluate(self.record(1, (1/3, 1/3, 1/3)), self.graph)
        self.assertIn('semantic_uncertainty', out['reasons'])
        self.assertTrue(out['need_more_observation'])
        self.assertNotIn('insufficient_view_diversity', out['reasons'])
    def test_consistent_repeated_view_does_not_request_other_angles(self):
        self.graph.poses[1] = np.eye(4)
        self.db.records.append(self.record(0))
        out = self.db.evaluate(self.record(1), self.graph)
        self.assertEqual(out['independent_views'], 1)
        self.assertFalse(out['need_more_observation'])
        self.assertEqual(out['status'], 'sufficient')
    def test_uncertain_first_frame_can_request_observation(self):
        out = self.db.evaluate(self.record(0, (1/3, 1/3, 1/3)), self.graph)
        self.assertTrue(out['need_more_observation'])
        self.assertEqual(out['reasons'], ['semantic_uncertainty'])
    def test_same_semantics_different_place_not_clustered(self):
        self.db.records.append(self.record(0))
        current = self.record(1)
        current['points'] += [0, 0, 4]
        out = self.db.evaluate(current, self.graph)
        self.assertEqual(out['scene_members'], [])
    def test_pose_correction_applies_to_old_observations(self):
        self.db.records.append(self.record(0))
        self.graph.poses[0][2, 3] = 3
        out = self.db.evaluate(self.record(1), self.graph)
        self.assertEqual(out['scene_members'], [])
    def test_database_artifact_and_overlap_dedup(self):
        source = Path(self.tmp.name) / 'semantic.npz'
        np.savez(source, pixel_indices=np.arange(64), probabilities=np.tile([.01, .98, .01], (64, 1)), class_ids=np.ones((8, 8), np.uint16))
        processor = SimpleNamespace(observation_paths={'frame_0.png': source})
        sm = SimpleNamespace(image_paths=['frame_0.png'], base_id=0, frame_ids=[0],
                             points=self.world.reshape(1, 8, 8, 3), conf=np.ones((1, 8, 8)), conf_threshold=.5,
                             colors=np.zeros((1,8,8,3),np.uint8), retrieval_vectors=np.array([[1.,0.]]),
                             mask=None, poses_local=np.eye(4)[None], intrinsics=np.eye(3)[None], key=lambda i: 0)
        first = self.db.add_submap(sm, self.graph, processor)
        self.assertEqual(len(first), 1)
        self.assertEqual(self.db.add_submap(sm, self.graph, processor), [])
        self.assertTrue((Path(self.tmp.name) / first[0]['artifact']).exists())
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0], 1)
        self.assertEqual(len((Path(self.tmp.name) / 'decisions.jsonl').read_text().splitlines()), 1)
        self.graph.poses[0][0, 3] = 2
        self.db.refresh_poses(self.graph)
        self.assertEqual(json.loads(self.db.db.execute('SELECT pose FROM frames').fetchone()[0])[0][3], 2)
    def test_insufficient_geometry(self):
        r = self.record(0); r['points'] = r['points'][:2]; r['probs'] = r['probs'][:2]
        self.assertEqual(self.db.evaluate(r, self.graph)['reasons'], ['insufficient_valid_geometry'])
        self.assertFalse(self.db.evaluate(r, self.graph)['need_more_observation'])
    def test_jsd_bounds(self):
        np.testing.assert_allclose(jsd([[1, 0]], [[0, 1]]), [1])
        np.testing.assert_allclose(jsd([[.5, .5]], [[.5, .5]]), [0])

    def test_nonconsecutive_old_frame_retrieved_beyond_nearest_cameras(self):
        self.db.cfg['retrieval_topk'] = 1
        self.db.records.append(self.record(0))
        for key in range(2, 32):
            self.graph.poses[key] = np.eye(4)
            distraction = self.record(key)
            distraction['descriptor'] = np.array([0., 1.])
            distraction['points'] += [0, 0, 4]
            self.db.records.append(distraction)
        # No radius or recency exclusion: camera has moved beyond the old 8 m gate.
        self.graph.poses[1][0, 3] = 9
        current = self.record(1)
        out = self.db.evaluate(current, self.graph)
        self.assertEqual([m['frame'] for m in out['scene_members']], ['frame0'])
        self.assertFalse(out['need_more_observation'])

    def test_visual_alias_with_no_shared_surface_rejected(self):
        self.db.records.append(self.record(0))
        query = self.record(1); query['points'] += [0, 0, 4]
        result = self.db.evaluate(query, self.graph)
        self.assertEqual(result['scene_members'], [])
        self.assertEqual(result['retrieval_candidates'][0]['verification'], 'no_shared_surface')

    def test_geometry_overlap_without_matching_visual_features_rejected(self):
        self.db.records.append(self.record(0))
        query = self.record(1)
        query['local_descriptors'] = np.random.default_rng(99).integers(0,256,(64,32),dtype=np.uint8)
        result = self.db.evaluate(query, self.graph)
        self.assertEqual(result['scene_members'], [])
        self.assertEqual(result['retrieval_candidates'][0]['verification'], 'local_visual_geometry_rejected')

    def test_anchor_not_hidden_by_recent_duplicate_images(self):
        index = VisualSceneIndex()
        index.add([1.,0.], anchor=True)
        for _ in range(30): index.add([1.,0.])
        hits = index.query([1.,0.], top_k=1, anchor_top_k=1, min_similarity=.3)
        self.assertIn(0, [hit['index'] for hit in hits])
        self.assertIn('group_anchor', next(hit for hit in hits if hit['index']==0)['sources'])

    def test_local_visual_matches_rejected_after_pose_misalignment(self):
        q, old = self.record(1), self.record(0)
        pairs, _, ratio = verified_local_pairs(q, old, self.world, self.world+[0,0,1], self.db.cfg)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(ratio, 0)

    def test_nonconsecutive_frames_share_persistent_group_across_submaps(self):
        source = Path(self.tmp.name) / 'semantic.npz'
        np.savez(source, pixel_indices=np.arange(64), probabilities=np.tile([.01,.98,.01],(64,1)), class_ids=np.ones((8,8),np.uint16))
        processor = SimpleNamespace(observation_paths={'frame_1.png':source,'frame_30.png':source})
        features = {key:self.record(0)[key] for key in ('local_descriptors','local_samples')}
        decisions = []
        with patch('ma_slam.reobserve.local_features',return_value=features):
            for key,frame_id in ((0,1),(1,30)):
                sm = SimpleNamespace(base_id=key,image_paths=[f'frame_{frame_id}.png'],frame_ids=[frame_id],
                                     points=self.world.reshape(1,8,8,3),conf=np.ones((1,8,8)),conf_threshold=.5,
                                     colors=np.zeros((1,8,8,3),np.uint8),retrieval_vectors=np.array([[1.,0.]]),
                                     mask=None,poses_local=self.graph.poses[key][None],intrinsics=np.eye(3)[None],key=lambda i,k=key:k)
                decisions.extend(self.db.add_submap(sm,self.graph,processor))
        self.assertEqual(decisions[0]['scene_group_ids'],decisions[1]['scene_group_ids'])
        self.assertTrue(decisions[1]['scene_members'][0]['frame'].endswith('frame_1.png'))
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM scene_groups').fetchone()[0],1)
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM scene_group_members').fetchone()[0],2)
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM scene_edges').fetchone()[0],1)
        with np.load(Path(self.tmp.name)/decisions[1]['artifact']) as saved:
            self.assertEqual(saved['descriptor'].shape,(2,))
            self.assertEqual(saved['local_samples'].shape,(64,))

    def test_three_factor_decision_and_resume_preserve_evidence(self):
        from ma_slam.region_evidence import RegionEvidence
        self.db.region_store=RegionEvidence(self.db.db)
        source=Path(self.tmp.name)/'semantic.npz'
        np.savez(source,pixel_indices=np.arange(64),probabilities=np.tile([.01,.98,.01],(64,1)),class_ids=np.ones((8,8),np.uint16))
        processor=SimpleNamespace(observation_paths={'frame_1.png':source,'frame_30.png':source})
        features={key:self.record(0)[key] for key in ('local_descriptors','local_samples')}
        outcomes=[]
        with patch('ma_slam.reobserve.local_features',return_value=features):
            for key,frame_id in ((0,1),(1,30)):
                sm=SimpleNamespace(base_id=key,image_paths=[f'frame_{frame_id}.png'],frame_ids=[frame_id],
                                   points=self.world.reshape(1,8,8,3),conf=np.ones((1,8,8)),conf_threshold=.5,
                                   colors=np.zeros((1,8,8,3),np.uint8),retrieval_vectors=np.array([[1.,0.]]),mask=None,
                                   poses_local=self.graph.poses[key][None],intrinsics=np.eye(3)[None],key=lambda i,k=key:k)
                outcomes.extend(self.db.add_submap(sm,self.graph,processor))
                if key==0:
                    self.db.close()
                    self.db=ObservationDatabase(self.tmp.name,dict(min_matches=5),region_config={'enable':True},resume=True)
        self.assertIs(outcomes[0]['need_more_observation'],True)
        self.assertIs(outcomes[1]['need_more_observation'],False)
        self.assertEqual(outcomes[1]['policy'],'semantic_geometry_regions_v6')
        self.assertTrue(outcomes[1]['region_decisions'][0]['semantic_evidence']['median_strength']>=1.5)
        self.assertEqual(self.db.db.execute('SELECT MIN(observations) FROM region_samples').fetchone()[0],2)

    def test_first_selection_ignores_other_submap_frames_and_semantics(self):
        self.db.frame_selection='first'
        source=Path(self.tmp.name)/'first.npz'
        np.savez(source,pixel_indices=np.arange(64),probabilities=np.tile([.01,.98,.01],(64,1)),class_ids=np.ones((8,8),np.uint16))
        processor=SimpleNamespace(observation_paths={'frame_0.png':source})
        provider=type('Provider',(),{'describe':lambda this,paths:np.array([[1.,0.]])})()
        self.db.descriptor_provider=provider
        sm=SimpleNamespace(base_id=0,image_paths=['frame_0.png','frame_1.png'],frame_ids=[0,1],
                           points=np.repeat(self.world.reshape(1,8,8,3),2,axis=0),conf=np.ones((2,8,8)),conf_threshold=.5,
                           colors=np.zeros((2,8,8,3),np.uint8),retrieval_vectors=None,mask=None,
                           poses_local=np.repeat(np.eye(4)[None],2,axis=0),intrinsics=np.repeat(np.eye(3)[None],2,axis=0),key=lambda i:i)
        result=self.db.add_submap(sm,self.graph,processor)
        self.assertEqual([r['frame_id'] for r in result],[0])
        self.assertEqual(result[0]['submap_frame_index'],0)
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0],1)
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM memberships').fetchone()[0],1)

    def test_semantic_sample_scores_align_with_dense_labels(self):
        semantic = OneFormerSemantic.__new__(OneFormerSemantic)
        semantic.device = 'cpu'
        semantic.store_observations = True
        semantic._load = lambda: None
        class Inputs(dict):
            def to(self, device): return self
        semantic.processor = lambda **kwargs: Inputs()
        logits = torch.full((1, 2, 103), -10.)
        logits[0, 0, 1] = 10
        logits[0, 1, 2] = 10
        masks = torch.tensor([[[[10., -10.], [10., -10.]],
                                [[-10., 10.], [-10., 10.]]]])
        semantic.model = lambda **kwargs: SimpleNamespace(class_queries_logits=logits, masks_queries_logits=masks)
        labels = semantic.infer_rgb(np.zeros((2, 2, 3), np.uint8))
        indices, probabilities = semantic._observation
        np.testing.assert_allclose(probabilities.sum(-1), 1, atol=.001)
        np.testing.assert_array_equal(probabilities.argmax(-1), labels.ravel()[indices])
        np.testing.assert_array_equal(labels, [[1, 2], [1, 2]])


if __name__ == '__main__':
    unittest.main()
