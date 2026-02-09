import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt

# 1. 환경 초기화
env = env_base('sim001.yaml')

# 중간 탐색 포인트 (Prey/Milestones)
target_points = np.array([
    [5, 5], [15, 5], [5, 11], [15, 11], 
    [10, 18], [10, 2]
])

# 로봇 객체에 기억 장치 주입
for robot in env.robot_list:
    robot.visited_points = set()

# 시뮬레이션 루프
for i in range(5000):
    vel_list = []
    
    for r_idx, robot in enumerate(env.robot_list):
        robot.cal_lidar_range(env.components) # 센서 업데이트
        pos = np.squeeze(robot.state[0:2])
        
        # 1) 포인트 방문 체크 (중간 과정)
        for p_idx, pt in enumerate(target_points):
            if np.linalg.norm(pos - pt) < 0.6:
                robot.visited_points.add(p_idx)

        # 2) 타겟 결정 로직
        target = None
        # 아직 수집 못한 포인트가 있다면 포인트 중 가장 가까운 곳 선택
        if len(robot.visited_points) < len(target_points):
            min_dist = float('inf')
            for p_idx, pt in enumerate(target_points):
                if p_idx not in robot.visited_points:
                    dist = np.linalg.norm(pos - pt)
                    if dist < min_dist:
                        min_dist, target = dist, pt
        else:
            # 모든 포인트를 다 알게 되면 최종 Goal 지점으로 타겟 설정
            # robot.goal은 YAML에서 정의된 값을 mobile_robot.py가 가지고 있음
            target = np.squeeze(robot.goal)

        # 3) 속도 계산 (APF 적용)
        if target is not None:
            f_att = (target - pos) / np.linalg.norm(target - pos)
            f_rep = np.array([0.0, 0.0])
            if robot.lidar is not None:
                for d, a in zip(robot.lidar.range_data, robot.lidar.angle_list):
                    if d < 1.8:
                        actual_a = robot.state[2, 0] + a
                        rep_dir = np.array([np.cos(actual_a), np.sin(actual_a)])
                        f_rep -= (rep_dir / (max(d, 0.1)**2))
            
            # 목표 지점에 가까워지면 감속 (도착 안정성)
            dist_to_target = np.linalg.norm(target - pos)
            speed = 1.2 if dist_to_target > 1.0 else 1.2 * dist_to_target
            vel = speed * f_att + 0.6 * f_rep
        else:
            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    # 정보 공유 (로봇 조우 시)
    if np.linalg.norm(np.squeeze(env.robot_list[0].state[0:2]) - np.squeeze(env.robot_list[1].state[0:2])) < 2.0:
        shared = env.robot_list[0].visited_points | env.robot_list[1].visited_points
        env.robot_list[0].visited_points = shared.copy()
        env.robot_list[1].visited_points = shared.copy()

    # 4. 시뮬레이션 업데이트
    env.robot_step(vel_list, vel_type='omni')
    env.render(0.001)



    # 추가: target_points(피식자) 시각화 로직
    for p_idx, pt in enumerate(target_points):
        # 팀 전체가 방문한 것을 알면 회색, 아니면 노란색 별
        is_visited = p_idx in env.robot_list[0].visited_points
        color = 'gray' if is_visited else 'yellow'
        env.world_plot.ax.plot(pt[0], pt[1], marker='o', color=color, markersize=8, markeredgecolor='black')
    plt.pause(0.01) # 실제 화면 갱신을 위한 대기


    plt.pause(0.01)

    # 종료 조건: 모든 로봇이 각자의 Goal에 도착했는지 체크
    if env.arrive_check():
        print("Mission Complete: All robots reached their final goals!")
        break

# env.end() 삭제 (AttributeError 방지)