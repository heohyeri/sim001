import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ir_sim.env import env_base

class MultiRobotSearchEnv(gym.Env):
    def __init__(self):
        super(MultiRobotSearchEnv, self).__init__()
        
        # 1. 환경 및 로봇 초기화
        self.env = env_base('sim001.yaml')
        self.num_robots = 2
        self.target_points = np.array([
            [1.5, 1.5], [8.5, 2.5], [5.0, 1.5], [1.5, 5.5], 
            [8.5, 4.5], [2.0, 9.0], [8.0, 9.0], [5, 6], [5, 9]
        ])
        
        # 2. 행동 공간 (Action Space): [선속도 v, 각속도 w]
        # vel_max [1.0, 1.0]에 맞춰 설정
        # self.action_space = spaces.Box(low=np.array([0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        
        # 3. 상태 공간 (Observation Space): LiDAR(30) + 목표물 상대 위치(2) = 32차원
        self.observation_space = spaces.Box(low=0, high=10.0, shape=(32,), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_robots, 2))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset() # 시뮬레이터 초기화
        
        for robot in self.env.robot_list:
            robot.visited_points = set()
        self.prev_actions = np.zeros((self.num_robots, 2))
        return self._get_obs(0), {} # 첫 번째 로봇 기준 상태 반환

    def _get_obs(self, r_idx):
        robot = self.env.robot_list[r_idx]
        robot.cal_lidar_range(self.env.components) # LiDAR 업데이트
        
        # 1. LiDAR 데이터 (정규화)
        lidar_data = np.array(robot.lidar.range_data) / 1.5 
        
        # 2. 가장 가까운 타겟 정보
        pos = np.squeeze(robot.state[0:2])
        target = self._get_closest_target(robot)
        rel_pos = (target - pos) if target is not None else np.array([0, 0])
        
        return np.concatenate([lidar_data, rel_pos]).astype(np.float32)

    def _get_closest_target(self, robot):
        # 1단계의 Greedy 타겟 선정 로직 재사용
        pos = np.squeeze(robot.state[0:2])
        min_dist = float('inf')
        closest = None
        for p_idx, pt in enumerate(self.target_points):
            if p_idx not in robot.visited_points:
                dist = np.linalg.norm(pos - pt)
                if dist < min_dist:
                    min_dist, closest = dist, pt
        return closest if closest is not None else pos

    def step(self, action_list):
        self.env.robot_step(action_list, vel_type='diff')
        self.env.collision_check()
        
        step_reward = 0
        terminated = False
        
        for r_idx, robot in enumerate(self.env.robot_list):
            pos = np.squeeze(robot.state[0:2])
            
            # 1. 포인트 방문 보상 (Sparse Reward)
            for p_idx, pt in enumerate(self.target_points):
                if np.linalg.norm(pos - pt) < 0.4 and p_idx not in robot.visited_points:
                    robot.visited_points.add(p_idx)
                    step_reward += 15.0 # 기존 10에서 상향

            # 2. 충돌 패널티 및 즉시 종료
            if robot.collision_flag:
                step_reward -= 10.0 # 기존 5에서 상향
                terminated = True
            
            # 3. 거리 기반 보상 (Dense Reward) - 타겟에 가까워지도록 유도
            target = self._get_closest_target(robot)
            dist = np.linalg.norm(pos - target)
            step_reward += 0.2 * (1.0 / (dist + 0.5)) # 거리가 가까울수록 보상 증가

            # 4. 주행 매끄러움 보상 (Smoothness) - 진동 문제 해결]
            # 이전 각속도와 차이가 크면 패널티 부여
            action_diff = np.abs(action_list[r_idx][1] - self.prev_actions[r_idx][1])
            step_reward -= 0.1 * action_diff
            self.prev_actions[r_idx] = action_list[r_idx]

            # 5. 생존 보상 - 단순히 벽을 피해 살아남는 것 자체에 보상
            if not robot.collision_flag:
                step_reward += 0.05

        # 정보 공유 로직
        if np.linalg.norm(np.squeeze(self.env.robot_list[0].state[0:2]) - np.squeeze(self.env.robot_list[1].state[0:2])) < 2.0:
            shared = self.env.robot_list[0].visited_points | self.env.robot_list[1].visited_points
            for r in self.env.robot_list: r.visited_points = shared.copy()

        if all(len(r.visited_points) == len(self.target_points) for r in self.env.robot_list):
            terminated = True
            step_reward += 30.0

        return self._get_obs(0), step_reward, terminated, False, {}

    def render(self):
        self.env.render(0.001)