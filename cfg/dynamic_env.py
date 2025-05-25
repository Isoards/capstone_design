import os
import gym
import traci
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

# SUMO configuration ---------------------------------------------------
SUMO_BINARY_GUI = "sumo-gui"       # For training/visualizing
SUMO_BINARY = "sumo"               # Headless binary (or same if no -gui suffix)
PROJECT_DIR = os.path.dirname(__file__)
SUMOCFG = os.path.join(PROJECT_DIR, "sejong.sumocfg")  # Path to your .sumocfg

# ID of the traffic light to control
TL_ID = "4821723515"

# Common SUMO command-line args
def _make_sumo_cmd(binary):
    return [
        binary,
        "-c", SUMOCFG,
        "--no-step-log",    # suppress step logs
        "--device.rerouting.probability", "0.0"  # optional: turn off dynamic rerouting
    ]

class DynamicSignalEnv(gym.Env):
    """
    Gym environment for a single intersection with dynamic discovery of
    incoming edges and number of phases.

    Observation: [q_in_0, ..., q_in_{N-1}, current_phase]
    Action     : Discrete(P)  → set phase index [0..P-1]
    Reward     : - mean(queue_lengths)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, sim_steps: int = 500, delta_time: float = 1.0):
        super().__init__()
        self.sim_steps = sim_steps
        self.delta_time = delta_time
        self.current_step = 0

        # --- Discover incoming edges and phase count via a headless SUMO run ---
        # ensure any existing connection is closed
        if traci.isLoaded():
            traci.close()
        # start SUMO in headless mode for parsing
        parse_cmd = _make_sumo_cmd(SUMO_BINARY)
        traci.start(parse_cmd)

        # get controlled links: list of [ (inLane, viaLane, outLane), ... ]
        links = traci.trafficlight.getControlledLinks(TL_ID)
        # flatten and extract unique incoming lane IDs
        in_lanes = [triple[0] for group in links for triple in group]
        # convert lane IDs to edge IDs by stripping lane suffix
        self.in_edges = sorted({lane.rsplit('_', 1)[0] for lane in in_lanes})

        # get phase definitions
        tl_defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(TL_ID)
        # assume only one program logic (index 0)
        phases = tl_defs[0].phases
        self.n_phases = len(phases)

        # close the parse session
        traci.close()

        # --- Define action and observation spaces dynamically ---
        self.action_space = spaces.Discrete(self.n_phases)
        # observation: queue for each incoming edge + current phase
        obs_dim = len(self.in_edges) + 1
        # allow large queue numbers, using inf as upper bound
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self):
        # start a fresh SUMO session with GUI (or headless if you prefer)
        if traci.isLoaded():
            traci.close()
        traci.start(_make_sumo_cmd(SUMO_BINARY_GUI))
        self.current_step = 0
        return self._get_state()

    def step(self, action: int):
        # apply the selected phase
        traci.trafficlight.setPhase(TL_ID, action)
        # advance the simulation
        traci.simulationStep()
        self.current_step += 1

        state  = self._get_state()
        reward = self._get_reward(state)
        done   = self.current_step >= self.sim_steps
        info   = {}

        if done:
            traci.close()
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        # fetch current queue length on each incoming edge
        queues = [traci.edge.getLastStepVehicleNumber(e) for e in self.in_edges]
        phase  = traci.trafficlight.getPhase(TL_ID)
        return np.array(queues + [phase], dtype=np.float32)

    @staticmethod
    def _get_reward(state: np.ndarray) -> float:
        # negative mean queue length → minimize queues
        return -float(np.mean(state[:-1]))

if __name__ == "__main__":
    # create environment and train PPO
    env = DynamicSignalEnv(sim_steps=1000)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(PROJECT_DIR, "tb_dynamic_tls")
    )
    model.learn(total_timesteps=200_000)
    model.save("ppo_dynamic_tls")

    # evaluate one episode
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
