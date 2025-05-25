#!/usr/bin/env python3
"""
교차로 10968712778 (incLanes = -896836750#1_0, 896836750#0_0)
를 대상으로 3 phase(0:G G / 1:y y / 2:r r) 중 하나를 선택하는
Discrete-action 환경을 구현한다.
"""
import os
import gym, traci
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

# ----------------------------------------------------------------------
# SUMO 실행 커맨드 ------------------------------------------------------
SUMO_BINARY = "sumo-gui"          # 학습 때는 "sumo" 로 바꿔도 됨
PROJECT_DIR = os.path.dirname(__file__)   # 스크립트 경로
SUMOCFG = os.path.join(PROJECT_DIR, "sejong.sumocfg")
#  sejong_traffic.sumocfg 안에는
#     <net-file value="../net/sejong_area.net.xml"/>
#     <route-files value="../routes/sejong.rou.xml"/>
#     <additional-files value="../taz/sejong_taz.add.xml"/>  가 들어있어야 합니다.

SUMO_CMD = [SUMO_BINARY, "-c", SUMOCFG]

# ----------------------------------------------------------------------
# 제어할 신호 정보 ------------------------------------------------------
TL_ID      = "10968712778"   # traffic-light id
# incLanes → edge id 두 개 (-896836750#1 , 896836750#0) :contentReference[oaicite:0]{index=0}
IN_EDGES   = ["-896836750#1", "896836750#0"]

# ----------------------------------------------------------------------
class SejongSignalEnv(gym.Env):
    """
    Observation : [q_edge0 , q_edge1 , phaseIndex]
    Action      : {0,1,2}  ← sejong_area.net.xml 에 정의된 3개 phase
    Reward      : − mean(queue)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, delta_time: float = 1.0, sim_steps: int = 500):
        super().__init__()
        self.delta_time  = delta_time
        self.sim_steps   = sim_steps
        self.current_step = 0

        # -------- spaces --------
        self.action_space      = spaces.Discrete(3)         # phase 0,1,2
        high = np.array([50, 50, 2], dtype=np.float32)      # 두 차로 + phase
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.float32)

    # --------------------------------------------------------------
    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(SUMO_CMD)
        self.current_step = 0
        return self._get_state()

    # --------------------------------------------------------------
    def step(self, action: int):
        # 1) 행동 적용 : 선택한 phase 로 전환
        traci.trafficlight.setPhase(TL_ID, int(action))

        # 2) 시뮬레이션 진행
        traci.simulationStep()
        self.current_step += 1

        # 3) 상태·보상·종료判定
        state  = self._get_state()
        reward = self._get_reward(state)
        done   = self.current_step >= self.sim_steps
        info   = {}

        if done:
            traci.close()
        return state, reward, done, info

    # --------------------------------------------------------------
    def _get_state(self):
        queues = [traci.edge.getLastStepVehicleNumber(e) for e in IN_EDGES]
        phase  = traci.trafficlight.getPhase(TL_ID)
        return np.array(queues + [phase], dtype=np.float32)

    # --------------------------------------------------------------
    @staticmethod
    def _get_reward(state: np.ndarray) -> float:
        """보상: 평균 대기열 길이 최소화"""
        return -float(np.mean(state[:-1]))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = SejongSignalEnv()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tb_sejong"
    )
    model.learn(total_timesteps=100_000)
    model.save("ppo_sejong_signal")

    # ---- 학습 후 one-shot 평가 ----
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
