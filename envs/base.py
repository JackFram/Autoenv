from envs.utils import dict_get, max_n_objects, fill_infos_cache, sample_trajdata_vehicle, load_ngsim_trajdatas
from src.Record.frame import Frame
from src.Record.record import SceneRecord, get_scene
from src.Basic.Vehicle import Vehicle
from feature_extractor.utils import build_feature_extractor
from envs.action import AccelTurnrate, propagate
from src.Vec.VecE2 import norm
import copy


class AutoEnv:
    def __init__(self, params: dict, trajdatas: list = None, trajinfos: list = None, roadways: list = None,
                 reclength: int = 5, delta_t: float = .1, primesteps: int = 50, H: int = 50,
                 terminate_on_collision: bool = True, terminate_on_off_road: bool = True,
                 render_params: dict = {"zoom": 5., "viz_dir": "/tmp"}):
        '''

        trajdatas::Vector{ListRecord}
        trajinfos::Vector{Dict}
        roadways::Vector{Roadway}
        roadway::Union{Nothing, Roadway} # current roadway
        scene::Scene
        rec::SceneRecord
        ext::MultiFeatureExtractor
        egoid::Int # current id of relevant ego vehicle
        ego_veh::Union{Nothing, Vehicle} # the ego vehicle
        traj_idx::Int # current index into trajdatas
        t::Int # current timestep in the trajdata
        h::Int # current maximum horizon for egoid
        H::Int # maximum horizon
        primesteps::Int # timesteps to prime the scene
        Δt::Float64

        # settings
        terminate_on_collision::Bool
        terminate_on_off_road::Bool

        # metadata
        epid::Int # episode id
        render_params::Dict # rendering options
        infos_cache::Dict # cache for infos intermediate results

        '''
        param_keys = params.keys()
        assert "trajectory_filepaths" in param_keys

        # optionally overwrite defaults
        reclength = dict_get(params, "reclength", reclength)
        primesteps = dict_get(params, "primesteps", primesteps)
        H = dict_get(params, "H", H)

        for (k, v) in dict_get(params, "render_params", render_params):
            render_params[k] = v

        terminate_on_collision = dict_get(params, "terminate_on_collision", terminate_on_collision)
        terminate_on_off_road = dict_get(params, "terminate_on_off_road", terminate_on_off_road)

        if trajdatas is None or trajinfos is None or roadways is None:
            trajdatas, trajinfos, roadways = load_ngsim_trajdatas(
                params["trajectory_filepaths"],
                minlength=primesteps + H
            )

        # build components
        scene_length = max_n_objects(trajdatas)
        scene = Frame()
        scene.init(scene_length)
        rec = SceneRecord()
        rec.init(reclength, delta_t, scene_length)
        ext = build_feature_extractor(params)
        infos_cache = fill_infos_cache(ext)

        self.trajdatas = trajdatas
        self.trajinfos = trajinfos
        self.roadways = roadways
        self.roadway = None
        self.scene = scene
        self.rec = rec
        self.ext = ext
        self.egoid = 0
        self.ego_veh = None
        self.traj_idx = 0
        self.t = 0
        self.h = 0
        self.H = H
        self.primesteps = primesteps
        self.delta_t = delta_t
        self.terminate_on_collision = terminate_on_collision
        self.terminate_on_off_road = terminate_on_off_road
        self.epid = 0
        self.render_params = render_params
        self.infos_cache = infos_cache

    def reset(self, offset: int = None, egoid: int = None, start: int = None, traj_idx: int = 1):
        if offset is None:
            offset = self.H + self.primesteps
        self.traj_idx, self.egoid, self.t, self.h = sample_trajdata_vehicle(
            self.trajinfos,
            offset,
            traj_idx,
            egoid,
            start
        )

        self.epid += 1

        self.rec.empty()
        self.scene.empty()

        # prime
        for t in range(self.t, (self.t + self.primesteps + 1)):
            self.scene = get_scene(self.scene, self.trajdatas[self.traj_idx], t)
            self.rec.update(self.scene)

        # set the ego vehicle
        self.ego_veh = self.scene[self.scene.findfirst(self.egoid)]
        # set the roadway
        self.roadway = self.roadways[self.traj_idx]
        # self.t is the next timestep to load
        self.t += self.primesteps + 1
        # enforce a maximum horizon 
        self.h = min(self.h, self.t + self.H)
        return self.get_features()

    def _step(self, action: list):
        # convert action into form
        # action[0] is `a::Float64` longitudinal acceleration [m/s^2]
        # action[1] is `ω::Float64` desired turn rate [rad/sec]
        ego_action = AccelTurnrate(action[0], action[1])

        # propagate the ego vehicle
        ego_state = propagate(
            self.ego_veh,
            ego_action,
            self.roadway,
            self.delta_t
        )
        # update the ego_veh
        self.ego_veh = Vehicle(ego_state, self.ego_veh.definition, self.ego_veh.id)

        # load the actual scene, and insert the vehicle into it
        self.scene = get_scene(self.scene, self.trajdatas[self.traj_idx], self.t)
        vehidx = self.scene.findfirst(self.egoid)
        orig_veh = self.scene[vehidx]  # for infos purposes
        self.scene[vehidx] = self.ego_veh

        # update rec with current scene
        self.rec.update(self.scene)

        # compute info about the step
        step_infos = dict()
        step_infos["rmse_pos"] = norm((orig_veh.state.posG - self.ego_veh.state.posG))
        step_infos["rmse_vel"] = norm((orig_veh.state.v - self.ego_veh.state.v))
        step_infos["rmse_t"] = norm((orig_veh.state.posF.t - self.ego_veh.state.posF.t))
        step_infos["x"] = self.ego_veh.state.posG.x
        step_infos["y"] = self.ego_veh.state.posG.y
        step_infos["s"] = self.ego_veh.state.posF.s
        step_infos["phi"] = self.ego_veh.state.posF.phi
        return step_infos

    def _extract_rewards(self, infos: dict):
        # rewards design
        r = 0
        if infos["is_colliding"] == 1:
            r -= 1
        if infos["is_offroad"] == 1:
            r -= 1
        return r

    def _compute_feature_infos(self, features: list):
        is_colliding = features[self.infos_cache["is_colliding_idx"]]
        markerdist_left = features[self.infos_cache["markerdist_left_idx"]]
        markerdist_right = features[self.infos_cache["markerdist_right_idx"]]
        is_offroad = features[self.infos_cache["out_of_lane_idx"]]
        return {
            "is_colliding": is_colliding,
            "markerdist_left": markerdist_left,
            "markerdist_right": markerdist_right,
            "is_offroad": is_offroad
        }

    def step(self, action: list):

        # action[0] is `a::Float64` longitudinal acceleration [m/s^2]
        # action[1] is `ω::Float64` desired turn rate [rad/sec]
        step_infos = self._step(action)

        # compute features and feature_infos
        features = self.get_features()
        feature_infos = self._compute_feature_infos(features)
        # combine infos
        infos = dict(**step_infos, **feature_infos)

        # update env timestep to be the next scene to load
        self.t += 1

        # compute terminal
        if self.t >= self.h:
            terminal = True
        elif self.terminate_on_collision and infos["is_colliding"] == 1:
            terminal = True
        elif self.terminate_on_off_road and (abs(infos["markerdist_left"]) > 3 and abs(infos["markerdist_right"]) > 3):
            terminal = True
        else:
            terminal = False

        reward = self._extract_rewards(infos)
        return features, reward, terminal, infos

    def get_features(self):
        veh_idx = self.scene.findfirst(self.egoid)
        self.ext.pull_features(
            self.rec,
            self.roadway,
            veh_idx
        )
        return copy.deepcopy(self.ext.features)

    def observation_space_spec(self):
        low = [0 for i in range(len(self.ext))]
        high = [0 for i in range(len(self.ext))]
        feature_infos = feature_info(self.ext)
        for (i, fn) in enumerate(self.ext.feature_names()):
            low[i] = feature_infos[fn]["low"]
            high[i] = feature_infos[fn]["high"]
        infos = {"high": high, "low": low}
        return (len(self.ext),), "Box", infos

    def action_space_spec(self):
        return (2,), "Box", {"high": [4., .15], "low": [-4., -.15]}

    def obs_names(self):
        return self.ext.feature_names()





