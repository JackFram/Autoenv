class WellBehavedFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(5)]
        self.num_features = 5

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        scene = rec[pastframe]
        veh_ego = scene[veh_idx]
        d_ml = get_markerdist_left(veh_ego, roadway)
        d_mr = get_markerdist_right(veh_ego, roadway)
        idx = 0
        self.features[idx] = convert(Float64, get(
            IS_COLLIDING, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert(Float64, d_ml < -1.0 or d_mr < -1.0)
        idx += 1
        self.features[idx] = convert(Float64, veh_ego.state.v < 0.0)
        idx += 1
        self.features[idx] = convert(Float64, get(
            ROADEDGEDIST_LEFT, rec, roadway, veh_idx, pastframe
        ))
        idx += 1
        self.features[idx] = convert(Float64, get(
            ROADEDGEDIST_RIGHT, rec, roadway, veh_idx, pastframe
        ))
        return self.features



