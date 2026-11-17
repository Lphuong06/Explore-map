"""CPU checks for label/geometry alignment and semantic propagation after pose changes."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ma_slam.submap import Submap
from ma_slam.map import GraphMap
from ma_slam.semantic import voxel_reduce,replica_palette
from model.ma_infer import MaChunkModel


class Graph:
    def get_pose(self,key):
        p=np.eye(4);p[0,3]=10
        return p


class SemanticTests(unittest.TestCase):
    def submap(self):
        sm=Submap(0);sm.n=1;sm.points=np.array([[[[1.,0,1],[2,0,1],[3,0,1],[4,0,1]]]])
        sm.colors=np.array([[[[10,20,30],[40,50,60],[70,80,90],[100,110,120]]]],np.uint8)
        sm.conf=np.ones((1,1,4));sm.conf_threshold=.5
        sm.poses_local=np.eye(4)[None];sm.poses_local[0,0,3]=2
        sm.semantic_labels=np.array([[[11,22,33,44]]],np.uint16)
        sm.mask=np.array([[[True,False,True,True]]]);return sm

    def test_pose_mask_range_and_sampling_keep_labels(self):
        sm=self.submap();p,c,label=sm.world_points_and_colors(Graph(),max_range=2,return_labels=True)
        np.testing.assert_array_equal(p[:,0],[9,11]);np.testing.assert_array_equal(label,[11,33]);np.testing.assert_array_equal(c[:,0],[10,70])
        p,c,label=sm.world_points_and_colors(Graph(),max_points=1,return_labels=True)
        self.assertEqual(len(p),1);self.assertEqual(int(label[0]),{9:11,11:33,12:44}[p[0,0]])

    def test_voxel_unknown_and_tie(self):
        p=np.array([[.01,0,0],[.02,0,0],[.03,0,0],[.04,0,0],[1,0,0]])
        c=np.full((5,3),100,np.uint8);labels=np.array([0,0,7,4,0],np.uint16)
        xyz,rgb,label=voxel_reduce(p,c,labels,1)
        np.testing.assert_array_equal(label,[4,0]);self.assertAlmostEqual(xyz[0,0],.025)

    def test_export_matching_geometry_palette_and_ids(self):
        gm=GraphMap();gm.add(self.submap())
        with tempfile.TemporaryDirectory() as d:
            gm.write_points(Graph(),str(Path(d)/'combined_pcd.ply'),max_points=0)
            def read(name):
                raw=(Path(d)/name).read_bytes();offset=raw.index(b'end_header\n')+11
                return np.frombuffer(raw,dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,)),('id','<u2')],offset=offset)
            rgb=read('combined_pcd.ply');sem=read('combined_semantic.ply')
            np.testing.assert_array_equal(rgb['xyz'],sem['xyz']);np.testing.assert_array_equal(rgb['id'],[11,33,44]);np.testing.assert_array_equal(sem['rgb'],replica_palette()[rgb['id']])

    def test_semantics_precede_geometry_forward(self):
        events=[]
        class Semantic:
            def annotate_views(self,views,paths):events.append('semantic');return torch.tensor([[[4,7]]])
        class Backbone:
            def infer(self,views,**kwargs):
                events.append('geometry')
                return [dict(conf=torch.ones(1,1,2),depth_z=torch.ones(1,1,2,1),camera_poses=torch.eye(4)[None],intrinsics=torch.eye(3)[None],pts3d=torch.ones(1,1,2,3),img_no_norm=torch.ones(1,1,2,3))]
        model=MaChunkModel.__new__(MaChunkModel);model.device='cpu';model.model=Backbone();model._base_infer_kwargs={}
        with patch('model.ma_infer.build_chunk_views',return_value=[{}]):
            out=model.infer_chunk(['frame_1.png'],semantic_processor=Semantic())
        self.assertEqual(events,['semantic','geometry']);self.assertEqual(tuple(out['semantic_labels'].shape),tuple(out['depth'].shape))
        self.assertEqual(out['semantic_labels'].tolist(),[[[4,7]]])
        events.clear()
        with patch('model.ma_infer.build_chunk_views',return_value=[{}]):
            baseline=model.infer_chunk(['frame_1.png'])
        self.assertEqual(events,['geometry'])
        self.assertNotIn('semantic_labels',baseline)
        torch.testing.assert_close(baseline['world_points'],out['world_points'])
        torch.testing.assert_close(baseline['images'],out['images'])

    def test_misaligned_labels_rejected(self):
        sm=self.submap();sm.semantic_labels=np.zeros((1,2,2),np.uint16)
        with self.assertRaises(ValueError):sm.world_points_and_colors(Graph(),return_labels=True)

    def test_first_frame_semantics_do_not_reduce_geometry_inputs(self):
        seen=[]
        class Semantic:
            frame_selection='first'
            def annotate_views(self,views,paths):
                seen.append(('semantic',len(views),list(paths)))
                return torch.tensor([[[4,7]]])
        class Backbone:
            def infer(self,views,**kwargs):
                seen.append(('geometry',len(views)))
                return [dict(conf=torch.ones(1,1,2),depth_z=torch.ones(1,1,2,1),camera_poses=torch.eye(4)[None],
                             intrinsics=torch.eye(3)[None],pts3d=torch.ones(1,1,2,3),img_no_norm=torch.ones(1,1,2,3)) for view in views]
        model=MaChunkModel.__new__(MaChunkModel);model.device='cpu';model.model=Backbone();model._base_infer_kwargs={}
        with patch('model.ma_infer.build_chunk_views',return_value=[{},{}]):
            output=model.infer_chunk(['frame_0.png','frame_1.png'],semantic_processor=Semantic())
        self.assertEqual(seen,[('semantic',1,['frame_0.png']),('geometry',2)])
        self.assertEqual(output['world_points'].shape[0],2)
        self.assertNotIn('semantic_labels',output)
        self.assertEqual(output['semantic_first_frame_labels'].tolist(),[[4,7]])

if __name__=='__main__':unittest.main()
