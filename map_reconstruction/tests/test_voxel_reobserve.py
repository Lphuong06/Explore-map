import json,sqlite3,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ma_slam.voxel_reobserve import VoxelObservationDatabase

class NoRetrievalSubmap(SimpleNamespace):
 @property
 def retrieval_vectors(self):raise AssertionError('Semantic voxel association must not request SALAD descriptors')

class VoxelEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.db=VoxelObservationDatabase(self.root/'db')
  self.graph=SimpleNamespace(poses={},get_pose=lambda key:self.graph.poses[key]);self.processor=SimpleNamespace(observation_paths={})
 def tearDown(self):
  self.db.close();self.tmp.cleanup()
 def add(self,fid,probs=None,target=None,pose=None,base=None):
  probs=np.array([.99,.005,.005]) if probs is None else np.asarray(probs)
  target=np.array([.02,.02,1.02]) if target is None else np.asarray(target)
  pose=np.eye(4) if pose is None else pose.copy();self.graph.poses[fid]=pose
  path=str(self.root/f'frame_{fid:05d}.png');source=self.root/f'sem_{fid}.npz'
  np.savez(source,pixel_indices=np.arange(64),probabilities=np.tile(probs,(64,1)),class_ids=np.ones((8,8),np.uint16))
  self.processor.observation_paths[path]=source
  sm=NoRetrievalSubmap(base_id=fid*10 if base is None else base,image_paths=[path],frame_ids=[fid],points=np.tile(target,(1,8,8,1)),conf=np.ones((1,8,8)),conf_threshold=.5,colors=np.zeros((1,8,8,3),np.uint8),poses_local=pose[None],mask=None,key=lambda i:fid)
  return self.db.add_submap(sm,self.graph,self.processor)
 def test_spatial_reuse_without_retrieval_or_frame_anchor(self):
  a=self.add(0)[0];pose=np.eye(4);pose[0,3]=.01;b=self.add(30,pose=pose)[0]
  self.assertEqual(a['new_voxels'],1);self.assertEqual(b['reused_voxels'],1)
  self.assertIs(b['need_more_observation'],False);self.assertFalse(b['image_retrieval_for_semantics']);self.assertFalse(b['frame_anchor_required'])
  self.assertEqual(len(self.db.voxels),1)
 def test_many_pixels_in_one_voxel_are_one_frame_of_evidence(self):
  row=self.add(0)[0];s=next(iter(self.db.voxels.values()))
  self.assertEqual(len(s['frames']),1);self.assertLessEqual(s['mass'],1.)
  self.assertIs(row['need_more_observation'],True);self.assertEqual(row['reasons'],['insufficient_semantic_evidence'])
 def test_overlap_does_not_double_count(self):
  self.add(0);mass=next(iter(self.db.voxels.values()))['mass'];self.assertEqual(self.add(0,base=100),[])
  self.assertEqual(next(iter(self.db.voxels.values()))['mass'],mass)
  self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0],1)
  self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM memberships').fetchone()[0],2)
 def test_same_voxel_semantic_conflict_is_detected(self):
  p=np.full(102,.01/101);p[0]=.99;q=np.roll(p,1)
  self.add(0,probs=p);r=self.add(1,probs=q)[0]
  self.assertEqual(r['reasons'],['semantic_inconsistency'])
  self.assertGreater(r['region_decisions'][0]['semantic_consistency']['jsd_p90'],.2)
 def test_different_world_voxels_are_not_semantic_conflict(self):
  self.add(0);r=self.add(1,probs=[.005,.99,.005],target=[2.02,.02,1.02])[0]
  self.assertEqual(r['reused_voxels'],0);self.assertEqual(r['reasons'],['insufficient_semantic_evidence'])
 def test_pose_change_removes_old_voxel_and_moves_evidence(self):
  self.add(0);oldkey=next(iter(self.db.voxels));mass=self.db.voxels[oldkey]['mass']
  self.graph.poses[0][0,3]=1.;self.db.refresh_poses(self.graph)
  self.assertNotIn(oldkey,self.db.voxels);key=next(iter(self.db.voxels));self.assertEqual(key[0],oldkey[0]+10)
  self.assertAlmostEqual(self.db.voxels[key]['mass'],mass);self.assertEqual(len(self.db.voxels[key]['frames']),1)
  self.db.refresh_poses(self.graph);self.assertAlmostEqual(self.db.voxels[key]['mass'],mass)
  self.assertEqual(self.db.db.execute('SELECT x FROM frame_voxels').fetchone()[0],key[0])
 def test_resume_reconstructs_same_evidence_after_pose_correction(self):
  self.add(0);self.add(1);self.graph.poses[0][0,3]=1.;self.db.refresh_poses(self.graph)
  before={k:(v['mass'],v['evidence'].copy(),set(v['frames'])) for k,v in self.db.voxels.items()}
  decisions=list(self.db.db.execute('SELECT decision FROM frames ORDER BY rowid'))
  self.db.close();self.db=VoxelObservationDatabase(self.root/'db',resume=True)
  self.assertEqual(set(before),set(self.db.voxels))
  for k,(mass,evidence,frames) in before.items():
   self.assertAlmostEqual(mass,self.db.voxels[k]['mass']);np.testing.assert_allclose(evidence,self.db.voxels[k]['evidence'],atol=1e-12);self.assertEqual(frames,self.db.voxels[k]['frames'])
  self.assertEqual(decisions,list(self.db.db.execute('SELECT decision FROM frames ORDER BY rowid')))
 def test_resume_rejects_different_voxel_size(self):
  self.add(0)
  with self.assertRaises(ValueError):VoxelObservationDatabase(self.root/'db',dict(voxel_size=.2),resume=True)
 def test_unobserved_directions_do_not_generate_heat(self):
  r=self.add(0)[0];region=r['region_decisions'][0]
  self.assertEqual(region['directional_heat_p90'],0.)
  self.assertEqual(region['semantic_consistency']['directional_comparable_voxels'],0)
  self.assertFalse(region['semantic_consistency']['missing_direction_penalty'])
 def test_class_space_changes_are_rejected(self):
  self.add(0)
  with self.assertRaises(ValueError):self.add(1,probs=[.99,.01])
 def test_multiple_view_bins_accumulate_without_missing_angle_rule(self):
  target=np.array([.05,.05,1.05])
  for fid,axis in enumerate([self.db.axes[0],self.db.axes[self.db.neighbors[0,0]]]):
   camera=target-axis*2.;forward=axis;right=np.cross([0.,1.,0.],forward);right/=np.linalg.norm(right);up=np.cross(forward,right)
   pose=np.eye(4);pose[:3,:3]=np.column_stack([right,up,forward]);pose[:3,3]=camera
   row=self.add(fid,probs=[.99,.005,.005] if fid==0 else [.005,.99,.005],target=target,pose=pose)[0]
  self.assertEqual(len(self.db.voxels),1);self.assertEqual(len(self.db.directions),2)
  g=row['region_decisions'][0];self.assertGreater(g['directional_heat_p90'],0.)
  self.assertEqual(g['semantic_consistency']['directional_comparable_voxels'],1)

if __name__=='__main__':unittest.main()
