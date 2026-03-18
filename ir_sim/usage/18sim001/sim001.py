import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt


env = env_base('sim001.yaml', figsize=(19.2, 19.2))
repulsion_range = 8.0
cruise_speed = 8.0
approach_gain = 1.0
render_interval = 5


def point_to_segment_distance(point, segment):
    start = np.array(segment[:2], dtype=float)
    end = np.array(segment[2:], dtype=float)
    direction = end - start
    length_sq = np.dot(direction, direction)

    if length_sq == 0:
        return np.linalg.norm(point - start)

    projection = np.dot(point - start, direction) / length_sq
    projection = np.clip(projection, 0.0, 1.0)
    closest = start + projection * direction
    return np.linalg.norm(point - closest)


def generate_target_points(
    count,
    world_size=(100, 100),
    margin=8.0,
    min_spacing=10.0,
    wall_clearance=4.0,
    seed=7
):
    rng = np.random.default_rng(seed)
    width, height = world_size
    points = []
    wall_segments = env.components['obs_lines'].obs_line_states

    while len(points) < count:
        candidate = np.array([
            rng.uniform(margin, width - margin),
            rng.uniform(margin, height - margin)
        ])

        if any(np.linalg.norm(candidate - existing) < min_spacing for existing in points):
            continue

        if any(point_to_segment_distance(candidate, segment) < wall_clearance for segment in wall_segments):
            continue

        points.append(candidate)

    return np.array(points)


target_points = generate_target_points(count=50)
target_colors = ['yellow'] * len(target_points)
target_plot = None


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
                    if d < repulsion_range:
                        actual_a = robot.state[2, 0] - np.pi / 2 + a
                        rep_dir = np.array([np.cos(actual_a), np.sin (actual_a)])
                        f_rep -= (rep_dir / (max(d, 0.1)**2))
            

            dist_to_target = np.linalg.norm(target - pos)
            speed = cruise_speed if dist_to_target > cruise_speed else approach_gain * dist_to_target
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

    if i % render_interval == 0:
        env.render(0.001)

        if target_plot is None:
            target_plot = env.world_plot.ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                s=25,
                c=target_colors,
                edgecolors='black'
            )

        for p_idx in range(len(target_points)):
            is_visited = any(p_idx in r.visited_points for r in env.robot_list)
            target_colors[p_idx] = 'gray' if is_visited else 'orange'

        target_plot.set_color(target_colors)

    if all(len(r.visited_points) == len(target_points) for r in env.robot_list):
        print(f"{i * 0.1:.1f} seconds: Mission Success! All prey captured through collaboration.")
        break
