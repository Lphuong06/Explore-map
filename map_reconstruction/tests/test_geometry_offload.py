"""Disk-backed submaps must export the same map and poses as in-memory mapping."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ma_slam.solver import MaSlam, DEFAULT_CONFIG


class SyntheticModel:
    def infer_chunk(self, paths, **kwargs):
        n = len(paths)
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, 0, 3] = np.arange(n)*0.1
        points = np.tile(np.array([[[0., 0., 1.], [.1, 0., 1.]],
                                   [[0., .1, 1.], [.1, .1, 1.]]]), (n, 1, 1, 1))
        points += poses[:, None, None, :3, 3]
        return dict(world_points=points, images=np.full((n, 2, 2, 3), 128, np.uint8),
                    world_points_conf=np.ones((n, 2, 2)), poses=poses,
                    depth=np.ones((n, 2, 2)), intrinsics=np.tile(np.eye(3), (n, 1, 1)))


class GeometryOffloadTests(unittest.TestCase):
    def test_disk_backed_mapping_matches_in_memory_export(self):
        import open3d as o3d
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = []
            for enabled in (False, True):
                config = copy.deepcopy(DEFAULT_CONFIG)
                config.update(submap_size=2, overlap=1, scale_correction=False,
                              offload_geometry=enabled)
                config['Loop']['enable'] = False
                config['Pointcloud']['max_range'] = 10.0
                slam = MaSlam(config=config, model=SyntheticModel(), device='cpu')
                out = root/str(enabled)
                slam.run([f'frame_{i:05d}.png' for i in range(5)], str(out),
                         mode='rgb+depth+intr', depth_paths=['synthetic']*5,
                         intrinsics=np.eye(3))
                maps.append(np.asarray(o3d.io.read_point_cloud(str(out/'combined_pcd.ply')).points))
                if enabled:
                    self.assertEqual(len(list((out/'submaps').glob('*.npz'))), 2)
                    self.assertTrue(all(sm.points is None for sm in slam.map.ordered()))
            np.testing.assert_allclose(maps[0], maps[1])
            self.assertEqual((root/'False/camera_poses.txt').read_text(),
                             (root/'True/camera_poses.txt').read_text())


if __name__ == '__main__':
    unittest.main()
