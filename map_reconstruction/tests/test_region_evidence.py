import json,sqlite3,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ma_slam.region_evidence import RegionEvidence,geometry_metrics
from ma_slam.submap import Submap

class RegionTests(unittest.TestCase):
 def setUp(self):
  self.db=sqlite3.connect(':memory:');self.engine=RegionEvidence(self.db)
 def tearDown(self):self.db.close()
 def states(self,mass=2,entropy=.1,jsd=.01):
  return [dict(mass=mass,entropy=entropy,jsd=jsd,observations=3) for _ in range(16)]
 def geometry(self):return dict(valid_fraction=.8,median_confidence_ratio=1.)
 def decide(self,states=None,geometry=None,association='verified_shared_surface',residual=.01):
  return self.engine._decide('region',self.states() if states is None else states,self.geometry() if geometry is None else geometry,association,residual)
 def test_supported_semantics_and_geometry_false(self):
  self.assertIs(self.decide()['need_more_observation'],False)
 def test_weak_evidence_true_without_missing_angle_rule(self):
  r=self.decide(self.states(mass=.5));self.assertIs(r['need_more_observation'],True)
  self.assertEqual(r['reasons'],['insufficient_semantic_evidence'])
 def test_semantic_conflict_true_with_good_geometry(self):
  self.assertEqual(self.decide(self.states(jsd=.7))['reasons'],['semantic_inconsistency'])
 def test_bad_geometry_does_not_assert_semantic_conflict(self):
  r=self.decide(self.states(jsd=.7),dict(valid_fraction=.01,median_confidence_ratio=1.))
  self.assertEqual(r['reasons'],['geometric_unreliability'])
 def test_unverified_region_is_null(self):
  self.assertIsNone(self.decide(association='unverified_association')['need_more_observation'])
 def test_missing_geometry_is_null(self):
  r=self.engine._decide('r',self.states(),None,'verified_shared_surface',.01)
  self.assertIsNone(r['need_more_observation'])
 def test_dirichlet_accumulates_observation_evidence(self):
  p=np.array([.99,.005,.005]);a=self.engine._update('r',1,p,1);b=self.engine._update('r',1,p,1)
  self.assertGreater(b['mass'],a['mass']);self.assertEqual(b['observations'],2)
  self.assertLess(b['entropy'],a['entropy'])
 def test_resume_keeps_dirichlet_strength(self):
  p=np.array([.99,.005,.005]);self.engine._update('r',1,p,1)
  restored=RegionEvidence(self.db);r=restored._update('r',1,p,1)
  self.assertEqual(r['observations'],2)
 def test_geometry_relative_confidence_scale_invariant(self):
  c=np.ones(100);mask=np.ones(100,bool)
  a=geometry_metrics(c,1,mask,mask);b=geometry_metrics(c*10,10,mask,mask)
  self.assertEqual(a['median_confidence_ratio'],b['median_confidence_ratio'])
 def test_offloaded_geometry_exports_and_releases(self):
  sm=Submap(0);sm.n=1;sm.points=np.array([[[[0.,0,1],[1,0,1]]]])
  sm.colors=np.zeros((1,1,2,3),np.uint8);sm.conf=np.ones((1,1,2));sm.conf_threshold=.5
  sm.poses_local=np.eye(4)[None]
  graph=type('Graph',(),{'get_pose':lambda self,key:np.eye(4)})()
  with tempfile.TemporaryDirectory() as directory:
   sm.offload_geometry(Path(directory)/'geometry.npz');self.assertIsNone(sm.points)
   points,_=sm.world_points_and_colors(graph);self.assertEqual(len(points),2);self.assertIsNone(sm.points)
   sm.load_geometry();self.assertEqual(sm.points.shape,(1,1,2,3))

if __name__=='__main__':unittest.main()

class ExplanationTests(unittest.TestCase):
 def test_explanation_uses_prior_posterior_and_does_not_change_decision(self):
  dbs=[sqlite3.connect(':memory:') for _ in range(2)]
  try:
   plain=RegionEvidence(dbs[0]);explained=RegionEvidence(dbs[1],explain=True)
   a=np.array([.99,.005,.005]);b=np.array([.005,.99,.005])
   for engine in (plain,explained):engine._update('r',0,a,1.)
   x=plain._update('r',0,b,1.);y=explained._update('r',0,b,1.)
   for key in ('mass','entropy','jsd','observations'):self.assertEqual(x[key],y[key])
   detail=y['explanation']
   self.assertEqual(detail['previous_top'][0]['class_id'],0)
   self.assertEqual(detail['current_top'][0]['class_id'],1)
   self.assertGreater(detail['jsd'],.2)
   self.assertEqual(detail['observations'],2)
  finally:
   for db in dbs:db.close()

 def test_explanation_preserves_verified_sample_correspondence(self):
  db=sqlite3.connect(':memory:')
  try:
   engine=RegionEvidence(db,explain=True)
   record=dict(frame='new',probs=np.array([[.01,.99],[.99,.01]]),
               geometry=dict(valid_fraction=.8,median_confidence_ratio=1.),geometry_weights=np.ones(2))
   old=dict(probs=np.array([[.99,.01],[.01,.99]]),geometry_weights=np.ones(2))
   decision=dict(scene_members=[dict(frame='anchor',sample_pairs=[(1,0)],world_residual_p90=.01)])
   result=engine.observe(record,decision,['r'],{'r':'anchor'},{'anchor':old})
   detail=result['region_decisions'][0]['sample_diagnostics'][0]
   self.assertEqual(detail['current_sample_index'],1)
   self.assertEqual(detail['anchor_sample_index'],0)
   self.assertEqual(result['region_decisions'][0]['anchor_frame'],'anchor')
   self.assertLess(detail['jsd'],.001)
  finally:db.close()


class SharedSurfaceGateTests(unittest.TestCase):
 def check_case(self,association,entropy=.9,mass=.1,jsd=.8,weak_geometry=False):
  db=sqlite3.connect(':memory:')
  try:
   engine=RegionEvidence(db)
   states=[dict(mass=mass,entropy=entropy,jsd=jsd,observations=1)]*16
   return engine._decide('r',states,dict(valid_fraction=.5,median_confidence_ratio=.2 if weak_geometry else 1.),association,None)
  finally:db.close()
 def test_first_frame_requests_more_semantic_evidence(self):
  r=self.check_case('first_observation')
  self.assertIs(r['need_more_observation'],True);self.assertIs(r['semantic_decision']['need_more_observation'],True)
  self.assertEqual(r['reasons'],['insufficient_semantic_evidence'])
  self.assertFalse(r['semantic_decision']['comparison_available'])
 def test_unverified_frame_cannot_request_even_with_high_semantic_entropy(self):
  r=self.check_case('unverified_association',mass=2.)
  self.assertIsNone(r['need_more_observation']);self.assertEqual(r['reasons'],['no_verified_shared_surface'])
 def test_geometry_can_request_while_semantic_is_unknown(self):
  r=self.check_case('unverified_association',mass=2.,weak_geometry=True)
  self.assertIs(r['need_more_observation'],True)
  self.assertIsNone(r['semantic_decision']['need_more_observation'])
  self.assertEqual(r['reasons'],['geometric_unreliability'])
 def test_verified_region_still_detects_uncertainty(self):
  r=self.check_case('verified_shared_surface')
  self.assertIs(r['semantic_decision']['need_more_observation'],True)
  self.assertEqual(r['reasons'],['semantic_uncertainty'])
 def test_verified_but_no_comparable_samples_is_unknown(self):
  r=self.check_case('verified_shared_surface',entropy=.1,mass=2.,jsd=None)
  self.assertIsNone(r['need_more_observation']);self.assertFalse(r['semantic_decision']['evaluable'])

 def test_unverified_low_evidence_requests_without_claiming_conflict(self):
  r=self.check_case('unverified_association',entropy=.9,mass=.3,jsd=.8)
  self.assertIs(r['need_more_observation'],True)
  self.assertEqual(r['reasons'],['insufficient_semantic_evidence'])
  self.assertFalse(r['semantic_decision']['comparison_available'])
 def test_both_semantic_and_geometry_reasons_are_retained(self):
  r=self.check_case('unverified_association',mass=.3,weak_geometry=True)
  self.assertIs(r['need_more_observation'],True)
  self.assertEqual(set(r['reasons']),{'insufficient_semantic_evidence','geometric_unreliability'})
 def test_shortage_still_requests_when_geometry_unavailable(self):
  db=sqlite3.connect(':memory:')
  try:
   r=RegionEvidence(db)._decide('r',[],None,'first_observation',None)
   self.assertIs(r['need_more_observation'],True)
   self.assertEqual(r['reasons'],['insufficient_semantic_evidence'])
  finally:db.close()
