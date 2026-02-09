import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt

env = env_base('sim001.yaml')

# 중간 탐색 포인트 (Prey)
# 로봇들이 최종 목적지에 가기 전 반드시 '수집'하거나 '방문'해야 하는 지점들입니다.
target_points = np.array([
    [5, 5], [15, 5], [5, 11], [15, 11], 
    [10, 18], [10, 2]
])

# 로봇 객체에 기억 장치 주입
# 로봇 객체에 'visited_points'라는 세트(Set)를 동적으로 추가
for robot in env.robot_list:
    robot.visited_points = set()

# 시뮬레이션 루프
for i in range(5000):
    vel_list = [] # 이번 스텝에서 로봇들이 움직일 속도 명령을 담을 리스트
    
    for r_idx, robot in enumerate(env.robot_list):
        robot.cal_lidar_range(env.components) # 센서 데이터 갱신: 현재 위치에서 주변 장애물(벽)과의 거리를 계산
        pos = np.squeeze(robot.state[0:2]) # 로봇의 현재 위치 (x, y)를 가져옴
        
        # 1) 포인트 방문 체크 (중간 과정)
        # 현재 위치와 각 포인트 사이의 거리를 계산하여 0.6m 이내면 방문한 것으로 간주합니다.
        for p_idx, pt in enumerate(target_points):
            if np.linalg.norm(pos - pt) < 0.6:
                robot.visited_points.add(p_idx)

        # 2) 타겟 결정 로직
        # 아직 수집 못한 포인트가 기억(visited_points)에 남아 있다면, 그중 가장 가까운 곳을 타겟으로 삼습니다.
        target = None
        if len(robot.visited_points) < len(target_points):
            min_dist = float('inf')
            for p_idx, pt in enumerate(target_points):
                if p_idx not in robot.visited_points:
                    dist = np.linalg.norm(pos - pt)
                    if dist < min_dist:
                        min_dist, target = dist, pt
        else:
            # 모든 포인트를 방문했다면, YAML에 설정된 로봇 각자의 최종 목적지(Goal)로 향합니다.
            target = np.squeeze(robot.goal)

        # 장애물 회피, Artificial Potential Field
        if target is not None:
            f_att = (target - pos) / np.linalg.norm(target - pos) # 인력(Attractive Force): 목표 지점으로 로봇을 끌어당기는 힘
            f_rep = np.array([0.0, 0.0]) # 척력(Repulsive Force): Lidar 센서가 감지한 벽으로부터 로봇을 밀어내는 힘
            if robot.lidar is not None:
                # Lidar의 각 레이저 빔(거리 d, 각도 a) 정보를 분석합니다.
                for d, a in zip(robot.lidar.range_data, robot.lidar.angle_list):
                    if d < 1.8: # 1.8m 이내에 벽이 감지되면
                        actual_a = robot.state[2, 0] + a # 로봇의 현재 방향을 고려하여 장애물의 실제 방향을 계산합니다.
                        rep_dir = np.array([np.cos(actual_a), np.sin(actual_a)])
                        f_rep -= (rep_dir / (max(d, 0.1)**2)) # 거리가 가까울수록 훨씬 강한 힘으로 밀어냅니다 (거리의 제곱에 반비례).
            
            # 목표 지점에 가까워지면 감속 (도착 안정성)
            dist_to_target = np.linalg.norm(target - pos)
            speed = 1.2 if dist_to_target > 1.0 else 1.2 * dist_to_target
            
            # 최종 속도 벡터 = 인력(목표 방향) + 척력(장애물 회피 방향)
            vel = speed * f_att + 0.6 * f_rep
        else:
            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    # 정보 공유 (로봇 조우 시)
    # 두 로봇 사이의 거리가 2.0m 이내가 되면 서로의 '방문 기록'을 합집합(Union) 연산하여 공유합니다.
    if np.linalg.norm(np.squeeze(env.robot_list[0].state[0:2]) - np.squeeze(env.robot_list[1].state[0:2])) < 2.0:
        shared = env.robot_list[0].visited_points | env.robot_list[1].visited_points
        env.robot_list[0].visited_points = shared.copy()
        env.robot_list[1].visited_points = shared.copy()

    # 시뮬레이션 상태 업데이트 및 화면 표시
    # 계산된 속도를 로봇들에게 전달하여 위치를 이동시킵니다.
    env.robot_step(vel_list, vel_type='omni')
    env.render(0.001)


    # target_points(피식자) 시각화 로직
    for p_idx, pt in enumerate(target_points):
        # 팀 전체가 방문 사실을 알고 있다면 회색, 아니면 노란색 원으로 표시
        is_visited = p_idx in env.robot_list[0].visited_points
        color = 'gray' if is_visited else 'yellow'
        env.world_plot.ax.plot(pt[0], pt[1], marker='o', color=color, markersize=8, markeredgecolor='black')
    
    plt.pause(0.01) # 애니메이션 효과를 위해 잠시 대기

    # 최종 종료 조건: 모든 로봇이 각자의 최종 Goal에 도달했는지 확인합니다.
    if env.arrive_check():
        print("Mission Complete: All robots reached their final goals!")
        break