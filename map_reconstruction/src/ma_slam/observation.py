"""Select semantic evidence storage independently of SLAM loop retrieval."""


def create_observation_database(root, config):
    settings = config.get("Reobserve", {})
    selection = settings.get("frame_selection", "all")
    association = settings.get("association", "world_voxel")
    if association == "world_voxel":
        from ma_slam.voxel_reobserve import VoxelObservationDatabase

        thresholds = dict(config.get("VoxelEvidence", {}))
        thresholds.update(settings.get("thresholds", {}))
        return VoxelObservationDatabase(
            root, config=thresholds, frame_selection=selection
        )
    if association == "legacy_region":
        from ma_slam.reobserve import ObservationDatabase

        return ObservationDatabase(
            root,
            settings.get("thresholds"),
            region_config=config.get("RegionDecision", {"enable": True}),
            frame_selection=selection,
        )
    raise ValueError(f"Unknown semantic association: {association}")
