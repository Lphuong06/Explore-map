"""Global visual retrieval and local appearance/3D verification for scene groups."""
import numpy as np
from scipy.spatial import cKDTree


class VisualSceneIndex:
    """Search all prior frames plus representative group anchors, without time/pose gates."""
    def __init__(self):
        self.index = None
        self.anchor_index = None
        self.anchors = []
        self.count = 0

    @staticmethod
    def vector(descriptor):
        vector = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if not len(vector) or not np.isfinite(vector).all() or norm <= 1e-12:
            raise ValueError('Visual descriptor must be finite and nonzero')
        return np.ascontiguousarray((vector / norm)[None])

    def add(self, descriptor, anchor=False):
        import faiss
        vector = self.vector(descriptor)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vector.shape[1])
            self.anchor_index = faiss.IndexFlatIP(vector.shape[1])
        if vector.shape[1] != self.index.d:
            raise ValueError('Visual descriptor dimensions changed')
        self.index.add(vector)
        if anchor:
            self.anchor_index.add(vector)
            self.anchors.append(self.count)
        self.count += 1

    def query(self, descriptor, top_k, anchor_top_k, min_similarity):
        vector = self.vector(descriptor)
        if self.index is None:
            return []
        if vector.shape[1] != self.index.d:
            raise ValueError('Visual descriptor dimensions changed')
        found = {}
        for index, count, source in ((self.index, top_k, 'frame'),
                                     (self.anchor_index, anchor_top_k, 'group_anchor')):
            if index.ntotal == 0:
                continue
            scores, ids = index.search(vector, min(int(count), index.ntotal))
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or score < min_similarity:
                    continue
                record_idx = self.anchors[int(idx)] if source == 'group_anchor' else int(idx)
                item = found.setdefault(record_idx, dict(index=record_idx, similarity=float(score), sources=[]))
                item['sources'].append(source)
        return sorted(found.values(), key=lambda item: (-item['similarity'], item['index']))


def local_features(rgb, sample_pixels, image_shape, max_features=2000, max_pixel_distance=6.0, fast_threshold=10):
    """ORB features linked to valid sampled 3D evidence on the exact MAV image grid."""
    import cv2
    rgb = np.asarray(rgb)
    if rgb.shape != (*image_shape, 3):
        raise ValueError('Local visual features must use the MAV geometry image grid')
    empty = dict(local_descriptors=np.empty((0, 32), np.uint8), local_samples=np.empty(0, np.int64))
    if len(sample_pixels) == 0:
        return empty
    image = np.ascontiguousarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))
    keypoints, descriptors = cv2.ORB_create(nfeatures=int(max_features), fastThreshold=int(fast_threshold)).detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    if descriptors is None:
        return empty
    coordinates = np.array([point.pt for point in keypoints])
    samples = np.column_stack([sample_pixels % image_shape[1], sample_pixels // image_shape[1]])
    distances, indices = cKDTree(samples).query(coordinates)
    keep = distances <= max_pixel_distance
    return dict(local_descriptors=descriptors[keep], local_samples=indices[keep].astype(np.int64))


def verified_local_pairs(query, old, query_world, old_world, cfg):
    """Mutual ratio-test ORB matches, verified against current MAV world geometry."""
    import cv2
    qd, od = query['local_descriptors'], old['local_descriptors']
    if min(len(qd), len(od)) < 2:
        return np.empty((0, 2), np.int64), 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def ratios(a, b):
        result = {}
        for pair in matcher.knnMatch(a, b, k=2):
            if len(pair) == 2:
                first, second = pair
                if first.distance < cfg['local_ratio'] * second.distance:
                    result[first.queryIdx] = first.trainIdx
        return result
    forward, backward = ratios(qd, od), ratios(od, qd)
    pairs = [(query['local_samples'][qi], old['local_samples'][oi])
             for qi, oi in forward.items() if backward.get(oi) == qi]
    if not pairs:
        return np.empty((0, 2), np.int64), 0, 0.0
    pairs = np.unique(np.asarray(pairs, np.int64), axis=0)
    # Enforce one-to-one sampled evidence as well as one-to-one image features.
    _, qi = np.unique(pairs[:, 0], return_index=True)
    pairs = pairs[qi]
    _, oi = np.unique(pairs[:, 1], return_index=True)
    pairs = pairs[oi]
    total = len(pairs)
    distances = np.linalg.norm(query_world[pairs[:, 0]] - old_world[pairs[:, 1]], axis=1)
    pairs = pairs[distances <= cfg['match_distance']]
    return pairs, total, len(pairs) / max(1, total)
