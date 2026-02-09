import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt

# 1. 시뮬레이션 환경 초기화
# YAML 파일을 읽어 지도와 로봇, Lidar 센서를 설정합니다.
env = env_base('sim001.yaml')

# 우리가 찾아야 할 피식자(Prey) 포인트들
# 이 포인트들을 모두 방문하고 정보를 공유하는 것이 이번 실험의 궁극적 목표입니다.
target_points = np.array([
    [1.5, 1.5], [8.5, 2.5], # 하단 구역 (좌/우)
    [5.0, 1.5],             # 하단 세로 벽 근처
    [1.5, 5.5], [8.5, 4.5], # 중간 통로 구역 (좌/우)
    [2.0, 9.0], [8.0, 9.0], [5,6], [5,9]# 상단 구석 구역 (좌/우 - 미로 깊숙한 곳)
])
# 로봇들에게 '기억 장치' 주입$
# 각 로봇은 자기가 직접 방문하거나 동료에게 전해 들은 포인트 번호를 세트(Set)에 저장합니다.
for robot in env.robot_list:
    robot.visited_points = set()

# 2. 메인 시뮬레이션 루프
for i in range(5000):
    vel_list = [] # 로봇들에게 줄 속도 명령 리스트
    
    for r_idx, robot in enumerate(env.robot_list):
        # 센서 업데이트: 주변 벽과의 거리를 측정하여 충돌을 방지합니다.
        robot.cal_lidar_range(env.components)
        pos = np.squeeze(robot.state[0:2])
        
        # [로직 1] 포인트 방문 체크 (사냥 성공 여부)
        # 로봇이 포인트 근처(0.6m)에 도달하면 해당 포인트를 수집한 것으로 간주합니다.
        for p_idx, pt in enumerate(target_points):
            if np.linalg.norm(pos - pt) < 0.4:
                robot.visited_points.add(p_idx)

        # [로직 2] 다음 타겟 결정 (Greedy 탐색)
        # 아직 수집하지 못한 포인트 중 '자신에게서 가장 가까운' 곳을 다음 목표로 정합니다.
        target = None
        if len(robot.visited_points) < len(target_points):
            min_dist = float('inf')
            for p_idx, pt in enumerate(target_points):
                if p_idx not in robot.visited_points:
                    dist = np.linalg.norm(pos - pt)
                    if dist < min_dist:
                        min_dist, target = dist, pt
        
        # [로직 3] 이동 제어 (장애물 회피 포함)
        if target is not None:
            # 인력(Attractive Force): 목표 지점으로 향하는 힘
            f_att = (target - pos) / np.linalg.norm(target - pos)
            
            # 척력(Repulsive Force): 벽에 부딪히지 않게 밀어내는 힘 (Lidar 데이터 활용)
            f_rep = np.array([0.0, 0.0])
            if robot.lidar is not None:
                for d, a in zip(robot.lidar.range_data, robot.lidar.angle_list):
                    if d < 1.8: # 1.8m 이내에 벽이 있으면 척력 발생
                        actual_a = robot.state[2, 0] - np.pi / 2 + a
                        rep_dir = np.array([np.cos(actual_a), np.sin (actual_a)])
                        f_rep -= (rep_dir / (max(d, 0.1)**2))
            
            # 도착 안정성을 위해 타겟에 가까워지면 속도를 줄입니다.
            dist_to_target = np.linalg.norm(target - pos)
            speed = 1.2 if dist_to_target > 1.0 else 1.2 * dist_to_target
            vel = speed * f_att + 0.6 * f_rep
        else:
            # 더 이상 갈 곳이 없으면 제자리에 멈춥니다.
            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    # [로직 4] 핵심: 정보 공유 (Strategic Rendezvous)
    # 두 로봇이 2.0m 이내로 가까워지면 서로가 가진 '피식자 정보'를 동기화합니다.
    # 이를 통해 로봇 2는 로봇 1이 이미 방문한 곳을 다시 갈 필요가 없음을 알게 됩니다.
    if np.linalg.norm(np.squeeze(env.robot_list[0].state[0:2]) - np.squeeze(env.robot_list[1].state[0:2])) < 2.0:
        # 합집합(|) 연산을 통해 정보를 합칩니다.
        shared = env.robot_list[0].visited_points | env.robot_list[1].visited_points
        env.robot_list[0].visited_points = shared.copy()
        env.robot_list[1].visited_points = shared.copy()

    # 3. 화면 렌더링 및 시각화
    env.robot_step(vel_list, vel_type='omni')
    env.collision_check()
    env.render(0.001)

    # 피식자(Prey) 표시: 방문하지 않은 곳은 노란색, 방문(인지)한 곳은 회색으로 변합니다.
    for p_idx, pt in enumerate(target_points):
        is_visited = any(p_idx in r.visited_points for r in env.robot_list)
        color = 'gray' if is_visited else 'yellow'
        env.world_plot.ax.plot(pt[0], pt[1], marker='o', color=color, markersize=8, markeredgecolor='black')
    
    plt.pause(0.01)

    # [로직 5] 최종 종료 조건 (Success Condition)
    # 모든 로봇이 모든 포인트의 위치와 방문 사실을 공유받았을 때 미션이 끝납니다.
    if all(len(r.visited_points) == len(target_points) for r in env.robot_list):
        print(f"{i * 0.1:.1f} seconds: Mission Success! All prey captured through collaboration.")
        break