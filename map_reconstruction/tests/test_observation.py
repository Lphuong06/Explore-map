"""The public pipeline must select the same evidence backend as voxel v2."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ma_slam.observation import create_observation_database
from ma_slam.voxel_reobserve import VoxelObservationDatabase


class ObservationBackendTests(unittest.TestCase):
    def test_default_uses_voxels_and_applies_threshold_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            db = create_observation_database(root, {
                'Reobserve': {'frame_selection': 'first',
                              'thresholds': {'voxel_size': 0.2}},
                'VoxelEvidence': {'voxel_size': 0.1, 'min_evidence': 2.0},
            })
            try:
                self.assertIsInstance(db, VoxelObservationDatabase)
                self.assertEqual(db.frame_selection, 'first')
                self.assertEqual(db.cfg['voxel_size'], 0.2)
                self.assertEqual(db.cfg['min_evidence'], 2.0)
            finally:
                db.close()

    def test_unknown_backend_fails_before_creating_data(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'Unknown semantic association'):
                create_observation_database(root, {'Reobserve': {'association': 'typo'}})
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
