from src.Roadway.roadway import Roadway


class ForeForeFeatureExtractor:
    def __init__(self, deltas_censor_hi: float = 100.):
        self.num_features = 3
        self.features = [0 for i in range(self.num_features)]
        self.deltas_censor_hi = deltas_censor_hi

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        # reset features
        self.features = [0 for i in range(self.num_features)]

        scene = rec[pastframe]

        ego_vel = scene[veh_idx].state.v

        vtpf = VehicleTargetPointFront()
        vtpr = VehicleTargetPointRear()
        fore_M = get_neighbor_fore_along_lane(scene, veh_idx, roadway, vtpf, vtpr, vtpf)
        if fore_M.ind != 0:
            fore_fore_M = get_neighbor_fore_along_lane(scene, fore_M.ind, roadway, vtpr, vtpf, vtpr)
        else:
            fore_fore_M = NeighborLongitudinalResult(0, 0.)

        if fore_fore_M.ind != 0:
            # total distance from ego vehicle
            self.features[0] = fore_fore_M.Δs + fore_M.Δs
            # relative velocity to ego vehicle
            self.features[1] = scene[fore_fore_M.ind].state.v - ego_vel
            # absolute acceleration
            self.features[2] = convert(Float64, get(ACCFS, rec, roadway, fore_fore_M.ind, pastframe))
        else:
            self.features[0] = self.deltas_censor_hi
            self.features[1] = 0.
            self.features[2] = 0.

        return self.features

