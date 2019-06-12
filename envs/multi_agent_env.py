from envs.utils import dict_get, max_n_objects, fill_infos_cache, sample_multiple_trajdata_vehicle, load_ngsim_trajdatas,\
    keep_vehicle_subset
from src.Record.frame import Frame
from src.Record.record import SceneRecord, get_scene
from src.Basic.Vehicle import Vehicle
from feature_extractor.utils import build_feature_extractor
from envs.action import AccelTurnrate, propagate
from src.Vec.VecE2 import norm
import copy
import numpy as np


class MultiAgentAutoEnv:
    def __init__(self, params: dict, trajdatas: list = None, trajinfos: list = None, roadways: list = None,
                 reclength: int = 5, delta_t: float = .1, primesteps: int = 50, H: int = 50,
                 n_veh: int = 20, remove_ngsim_veh: bool = False, reward: int = 0,
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
        n_veh = dict_get(params, "n_veh", n_veh)
        remove_ngsim_veh = dict_get(params, "remove_ngsim_veh", remove_ngsim_veh)
        reward = dict_get(params, "reward", reward)

        for (k, v) in dict_get(params, "render_params", render_params).items():
            render_params[k] = v

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

        # features are stored in row-major order because they will be transferred
        # to python; this is inefficient in julia, but don't change or it will
        # break the python side of the interaction
        features = [[0 for _ in range(len(ext))] for _ in range(n_veh)]
        egoids = [-1 for _ in range(n_veh)]
        ego_vehs = [None for _ in range(n_veh)]

        self.trajdatas = trajdatas
        self.trajinfos = trajinfos
        self.roadways = roadways
        self.roadway = None
        self.scene = scene
        self.rec = rec
        self.ext = ext
        self.egoids = egoids
        self.ego_vehs = ego_vehs
        self.traj_idx = 0
        self.t = 0
        self.h = 0
        self.H = H
        self.primesteps = primesteps
        self.delta_t = delta_t
        self.n_veh = n_veh
        self.remove_ngsim_veh = remove_ngsim_veh
        self.features = features
        self.reward = reward
        self.epid = 0
        self.render_params = render_params
        self.infos_cache = infos_cache


    '''
    
    Description:
    Reset the environment. Note that this environment maintains the following
    invariant attribute: at any given point, all vehicles currently being controlled
    will end their episode at the same time. This simplifies the rest of the code
    by enforcing synchronized restarts, but it does somewhat limit the sets of
    possible vehicles that can simultaneously interact. With a small enough minimum
    horizon (H <= 250 = 25 seconds) and number of vehicle (n_veh <= 100)
    this should not be a problem. If you need to run with larger numbers then those
    implement an environment with asynchronous resets.
    Args:
    - env: env to reset
    - dones: bool vector indicating which indices have reached a terminal state
        these must be either all true or all false
        
    '''

    def reset(self, dones: list = None, offset: int = None, random_seed: int = None,
              egoid: int = None, traj_idx: int = None, start: int = None):
        if offset is None:
            offset = self.H + self.primesteps
        if dones is None:
            dones = [True for _ in range(self.n_veh)]
        # enforce environment invariant reset property
        # (i.e., always either all true or all false)
        assert len(set(dones)) == 1
        # first == all at this point, so if first is false, skip the reset
        if not dones[0]:
            return

        # sample multiple ego vehicles
        # as stated above, these will all end at the same timestep
        self.traj_idx, self.egoids, self.t, self.h = sample_multiple_trajdata_vehicle(
            self.n_veh,
            self.trajinfos,
            offset,
            rseed=random_seed
        )

        self.epid += 1

        self.rec.empty()
        self.scene.empty()

        # prime
        for t in range(self.t, (self.t + self.primesteps + 1)):
            self.scene = get_scene(self.scene, self.trajdatas[self.traj_idx], t)
            if self.remove_ngsim_veh:
                self.scene = keep_vehicle_subset(self.scene, self.egoids)
            self.rec.update(self.scene)

        # set the ego vehicle
        for (i, egoid) in enumerate(self.egoids):
            vehidx = self.scene.findfirst(egoid)
            self.ego_vehs[i] = self.scene[vehidx]

        # set the roadway
        self.roadway = self.roadways[self.traj_idx]
        # self.t is the next timestep to load
        self.t += self.primesteps + 1
        # enforce a maximum horizon
        self.h = min(self.h, self.t + self.H)
        return self.get_features()

    def _step(self, action: list):
        # make sure number of actions passed in equals number of vehicles
        assert len(action) == self.n_veh
        ego_states = [None for _ in range(self.n_veh)]

        for (i, ego_veh) in enumerate(self.ego_vehs):
            # convert action into form
            ego_action = AccelTurnrate(action[i][0], action[i][1])
            # propagate the ego vehicle
            ego_state = propagate(
                ego_veh,
                ego_action,
                self.roadway,
                self.delta_t
            )
            # update the ego_veh
            self.ego_vehs[i] = Vehicle(ego_state, ego_veh.definition, ego_veh.id)

        # load the actual scene, and insert the vehicle into it
        self.scene = get_scene(self.scene, self.trajdatas[self.traj_idx], self.t)
        if self.remove_ngsim_veh:
            self.scene = keep_vehicle_subset(self.scene, self.egoids)

        orig_vehs = [None for _ in range(self.n_veh)]  # for infos purposes

        for (i, egoid) in enumerate(self.egoids):
            vehidx = self.scene.findfirst(egoid)

            # track the original vehicle for validation / infos purposes
            orig_vehs[i] = self.scene[vehidx]

            # replace the original with the controlled vehicle
            self.scene[vehidx] = self.ego_vehs[i]

        # update rec with current scene
        self.rec.update(self.scene)

        # compute info about the step
        step_infos = {
            "rmse_pos": [],
            "rmse_vel": [],
            "rmse_t": [],
            "x": [],
            "y": [],
            "s": [],
            "phi": [],
            "orig_x": [],
            "orig_y": [],
            "orig_theta": [],
            "orig_length": [],
            "orig_width": []
        }

        for i in range(self.n_veh):
            step_infos["rmse_pos"].append(norm((orig_vehs[i].state.posG - self.ego_vehs[i].state.posG)))
            step_infos["rmse_vel"].append(abs(orig_vehs[i].state.v - self.ego_vehs[i].state.v))
            step_infos["rmse_t"].append(abs((orig_vehs[i].state.posF.t - self.ego_vehs[i].state.posF.t)))
            step_infos["x"].append(self.ego_vehs[i].state.posG.x)
            step_infos["y"].append(self.ego_vehs[i].state.posG.y)
            step_infos["s"].append(self.ego_vehs[i].state.posF.s)
            step_infos["phi"].append(self.ego_vehs[i].state.posF.phi)
            step_infos["orig_x"].append(orig_vehs[i].state.posG.x)
            step_infos["orig_y"].append(orig_vehs[i].state.posG.y)
            step_infos["orig_theta"].append(orig_vehs[i].state.posG.theta)
            step_infos["orig_length"].append(orig_vehs[i].definition.length_)
            step_infos["orig_width"].append(orig_vehs[i].definition.width_)

        return step_infos

    def _extract_rewards(self, infos: dict):
        # rewards design
        rewards = [0 for _ in range(self.n_veh)]
        for i in range(self.n_veh):
            reward_col = infos["is_colliding"][i] * self.reward
            reward_off = infos["is_offroad"][i] * self.reward
            reward_brake = infos["hard_brake"][i] * self.reward * 0.5  # braking hard is not as bad as a collision
            rewards[i] = -max(reward_col, reward_off, reward_brake)
        return np.array(rewards)

    def _compute_feature_infos(self, features: list, accel_thresh_min: float=-2.0, accel_thresh: float = -3.0,
                               min_d_edge_thresh: float = 0.5, offroad_thresh: float = -0.1):
        feature_infos = {
            "is_colliding": [],
            "is_offroad": [],
            "hard_brake": []
        }
        for i in range(self.n_veh):
            is_colliding = features[i][self.infos_cache["is_colliding_idx"]]
            feature_infos["is_colliding"].append(is_colliding)
            accel = features[i][self.infos_cache["accel_idx"]]
            if accel <= accel_thresh_min:
                normalized_accel = abs((accel - accel_thresh_min) / (accel_thresh - accel_thresh_min))
                # linearly increase penalty up to accel_thresh
                feature_infos["hard_brake"].append(min(1, normalized_accel))
            else:
                feature_infos["hard_brake"].append(0)

            is_offroad = features[i][self.infos_cache["out_of_lane_idx"]]
            if is_offroad < 1:
                d_left = features[i][self.infos_cache["distance_road_edge_left_idx"]]
                d_right = features[i][self.infos_cache["distance_road_edge_right_idx"]]
                closest_d = min(d_left, d_right)
                if closest_d <= min_d_edge_thresh:  # meaning too close
                    is_offroad = abs((closest_d - min_d_edge_thresh) / (offroad_thresh - min_d_edge_thresh))
            feature_infos["is_offroad"].append(is_offroad)
        # print(feature_infos)

        return feature_infos

    def step(self, action: list):

        # action is a list of actions, length equals to self.n_veh
        step_infos = self._step(action)

        # compute features and feature_infos
        features = self.get_features()
        feature_infos = self._compute_feature_infos(list(features))

        # combine infos
        infos = dict(**step_infos, **feature_infos)

        # update env timestep to be the next scene to load
        self.t += 1

        # compute terminal
        if self.t > self.h:
            terminal = True
        else:
            terminal = False

        terminal = [terminal for _ in range(self.n_veh)]

        # vectorized sampler does not call reset on the environment
        # but expects the environment to handle resetting, so do that here
        # note: this mutates env.features in order to return the correct obs when resetting
        self.reset(terminal)

        rewards = self._extract_rewards(infos)
        # print(rewards)
        return copy.deepcopy(features), rewards, terminal, infos

    def get_features(self):
        for (i, egoid) in enumerate(self.egoids):
            veh_idx = self.scene.findfirst(egoid)
            self.ext.pull_features(self.rec, self.roadway, veh_idx)
            self.features[i] = copy.deepcopy(self.ext.features)
        return np.array(copy.deepcopy(self.features))

    def observation_space_spec(self):
        low = [0 for i in range(len(self.ext))]
        high = [0 for i in range(len(self.ext))]
        feature_infos = self.ext.feature_info()
        for (i, fn) in enumerate(self.ext.feature_names()):
            low[i] = feature_infos[fn]["low"]
            high[i] = feature_infos[fn]["high"]
        infos = {"high": high, "low": low}
        return (len(self.ext),), "Box", infos

    def action_space_spec(self):
        return (2,), "Box", {"high": [4., .15], "low": [-4., -.15]}

    def obs_names(self):
        return self.ext.feature_names()

    def vectorized(self):
        return True

    def num_envs(self):
        return self.n_veh

    @property
    def action_space(self):
        _, _, action_space = self.action_space_spec()
        return action_space





