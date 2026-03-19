import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt
from ir_sim.util.range_detection import range_seg_matrix, range_seg_seg


env = env_base('sim001.yaml', figsize=(19.2, 19.2))
repulsion_range = 8.0
cruise_speed = 12.0
approach_gain = 1.5
render_interval = 2

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
    seed=None
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


def to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def robot_can_detect_robot(observer, target, components):
    if observer.lidar is None:
        return False

    observer_pos = np.squeeze(observer.state[0:2])
    target_pos = np.squeeze(target.state[0:2])
    relative = target_pos - observer_pos
    distance = np.linalg.norm(relative)

    if distance == 0 or distance > observer.lidar.range_max:
        return False

    heading = observer.state[2, 0] - np.pi / 2
    bearing = np.arctan2(relative[1], relative[0]) - heading
    bearing = to_pi(bearing)

    if not (observer.lidar.angle_min <= bearing <= observer.lidar.angle_max):
        return False

    segment = [observer_pos, target_pos]
    blocked_by_map, _, map_range = range_seg_matrix(
        segment,
        components['map_matrix'],
        components['xy_reso'],
        observer.lidar.point_step_weight,
        components['offset']
    )

    if blocked_by_map and map_range < distance:
        return False

    for line in components['obs_lines'].obs_line_states:
        wall_segment = [
            np.array([line[0], line[1]], dtype=float),
            np.array([line[2], line[3]], dtype=float)
        ]
        blocked_by_wall, _, wall_range = range_seg_seg(segment, wall_segment)
        if blocked_by_wall and wall_range < distance:
            return False

    return True


def share_rendezvous_information(robot_list, components):
    robot_count = len(robot_list)
    visited = [False] * robot_count
    rendezvous_logs = []

    for start_idx in range(robot_count):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        component = []
        shared_points = set()

        while stack:
            idx = stack.pop()
            if visited[idx]:
                continue

            visited[idx] = True
            component.append(idx)
            shared_points |= robot_list[idx].visited_points

            for next_idx in range(robot_count):
                if visited[next_idx] or next_idx == idx:
                    continue

                if (
                    robot_can_detect_robot(robot_list[idx], robot_list[next_idx], components)
                    or robot_can_detect_robot(robot_list[next_idx], robot_list[idx], components)
                ):
                    stack.append(next_idx)

        component_changed = False
        for idx in component:
            if robot_list[idx].visited_points != shared_points:
                component_changed = True
            robot_list[idx].visited_points = shared_points.copy()

        if len(component) > 1 and component_changed:
            rendezvous_logs.append((sorted(component), len(shared_points)))

    return rendezvous_logs


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

    rendezvous_logs = share_rendezvous_information(env.robot_list, env.components)
    for component, shared_count in rendezvous_logs:
        print(
            f"{i * env.step_time:.1f}s rendezvous {component} "
            f"shared {shared_count}/{len(target_points)} targets"
        )

    env.robot_step(vel_list, vel_type='omni', stop=False)
    env.collision_check()

    if i % render_interval == 0:
        env.render(0.001, show_goal=False, show_text=False)

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
