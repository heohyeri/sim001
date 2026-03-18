import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt


env = env_base('sim001.yaml', figsize=(19.2, 19.2))
scale = 3.0

target_points = np.array([
    [1.5, 1.5], [8.5, 2.5],
    [5.0, 1.5],
    [1.5, 5.5], [8.5, 4.5],
    [2.0, 9.0], [8.0, 9.0], [5,6], [5,9]
]) * scale


for robot in env.robot_list:
    robot.visited_points = set()


for i in range(15000):
    vel_list = []
    
    for r_idx, robot in enumerate(env.robot_list):

        robot.cal_lidar_range(env.components)
        pos = np.squeeze(robot.state[0:2])
        

        for p_idx, pt in enumerate(target_points):
            if np.linalg.norm(pos - pt) < 1.2:
                robot.visited_points.add(p_idx)


        target = None
        if len(robot.visited_points) < len(target_points):
            min_dist = float('inf')
            for p_idx, pt in enumerate(target_points):
                if p_idx not in robot.visited_points:
                    dist = np.linalg.norm(pos - pt)
                    if dist < min_dist:
                        min_dist, target = dist, pt
        

        if target is not None:

            f_att = (target - pos) / np.linalg.norm(target - pos)

            f_rep = np.array([0.0, 0.0])
            if robot.lidar is not None:
                for d, a in zip(robot.lidar.range_data, robot.lidar.angle_list):
                    if d < 3.0:
                        actual_a = robot.state[2, 0] - np.pi / 2 + a
                        rep_dir = np.array([np.cos(actual_a), np.sin (actual_a)])
                        f_rep -= (rep_dir / (max(d, 0.1)**2))
            

            dist_to_target = np.linalg.norm(target - pos)
            speed = 6.0 if dist_to_target > 5.0 else 1.2 * dist_to_target
            vel = speed * f_att + 0.6 * f_rep
        else:

            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    if np.linalg.norm(np.squeeze(env.robot_list[0].state[0:2]) - np.squeeze(env.robot_list[1].state[0:2])) < 6.0:
        shared = env.robot_list[0].visited_points | env.robot_list[1].visited_points
        env.robot_list[0].visited_points = shared.copy()
        env.robot_list[1].visited_points = shared.copy()

    env.robot_step(vel_list, vel_type='omni')
    env.collision_check()
    env.render(0.001)

    for p_idx, pt in enumerate(target_points):
        is_visited = any(p_idx in r.visited_points for r in env.robot_list)
        color = 'gray' if is_visited else 'yellow'
        env.world_plot.ax.plot(pt[0], pt[1], marker='o', color=color, markersize=8, markeredgecolor='black')
    
    plt.pause(0.01)

    if all(len(r.visited_points) == len(target_points) for r in env.robot_list):
        print(f"{i * 0.1:.1f} seconds: Mission Success! All prey captured through collaboration.")
        break
