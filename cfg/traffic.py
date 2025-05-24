import os, gym, traci
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

SUMO_CMD = ["sumo-gui", "-c", "traffic.sumocfg"]

class SumoSignalEnv(gym.Env):
    def __init__(self, delta_time=1.0, sim_steps=500):
        super().__init__()
        self.delta_time = delta_time      # 시뮬레이션 한 스텝 길이
        self.sim_steps = sim_steps        # 에피소드 길이
        self.current_step = 0

        # 행동: 0=동서 초록 유지, 1=남북 초록 유지
        self.action_space = spaces.Discrete(2)

        # 상태: 각 방향 큐 길이(차량 수) + 현재 Phase(0 또는 1)
        high = np.array([50, 50, 50, 50, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.float32)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(SUMO_CMD)
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # 1) 행동 적용
        tl_id = "n2"
        if action == 0:
            traci.trafficlight.setPhase(tl_id, 0)   # 동서 방향 초록
        else:
            traci.trafficlight.setPhase(tl_id, 2)   # 남북 방향 초록

        # 2) 시뮬레이션 진행
        traci.simulationStep()      # delta_time = 1 s 기본
        self.current_step += 1

        # 3) 상태‧보상 계산
        state = self._get_state()
        reward = self._get_reward(state)
        done = self.current_step >= self.sim_steps
        info = {}

        if done:
            traci.close()
        return state, reward, done, info

    def _get_state(self):
        # 큐 길이: 각 도로 끝단 edge 차량 수
        north_q = traci.edge.getLastStepVehicleNumber("n2n5")
        south_q = traci.edge.getLastStepVehicleNumber("n4n2")
        east_q  = traci.edge.getLastStepVehicleNumber("n1n2")
        west_q  = traci.edge.getLastStepVehicleNumber("n2n3")
        phase = traci.trafficlight.getPhase("n2")  # 또는 실제 ID (확인 필요)
        return np.array([north_q, south_q, east_q, west_q, phase], dtype=np.float32)

    def _get_reward(self, state):
        # 평균 큐 길이를 최소화하는 보상 예시
        avg_q = np.mean(state[:4])
        return -avg_q      # 큐가 짧을수록 보상 ↑

# 환경 인스턴스 생성
env = SumoSignalEnv()

# PPO 학습
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb")
model.learn(total_timesteps=100_000)
model.save("ppo_basic_signal")

# 학습 후 평가
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
