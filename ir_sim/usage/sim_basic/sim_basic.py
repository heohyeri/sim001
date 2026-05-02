import numpy as np
import time
from ir_sim.env import env_base
import matplotlib.pyplot as plt
from ir_sim.util.range_detection import range_seg_matrix, range_seg_seg


env = env_base('sim_basic.yaml', figsize=(19.2, 19.2))
repulsion_range = 8.0
cruise_speed = 12.0
approach_gain = 1.5
render_interval = 2
repulsion_gain = 1.4
tangential_gain = 0.9
slowdown_range = 4.0
min_escape_speed = 1.2
escape_distance = 1.8
escape_steps = 10
escape_turn_gain = 1.8
backoff_gain = 1.1

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
    world_size=(180, 180),
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
    if observer.lidar is None or target.lidar is None:
        return False

    observer_pos = np.squeeze(observer.state[0:2])
    target_pos = np.squeeze(target.state[0:2])
    relative = target_pos - observer_pos
    distance = np.linalg.norm(relative)
    communication_distance = observer.lidar.range_max + target.lidar.range_max

    if distance == 0 or distance > communication_distance:
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


def sample_patrol_point(world_size=(180, 180), margin=8.0):
    while True:
        candidate = generate_target_points(
            count=1,
            world_size=world_size,
            margin=margin,
            min_spacing=0.0,
            wall_clearance=4.0,
            seed=None
        )[0]
        return candidate


def compute_avoidance_velocity(robot, target, pos):
    target_offset = target - pos
    dist_to_target = np.linalg.norm(target_offset)

    if dist_to_target < 1e-6:
        return np.zeros(2)

    f_att = target_offset / dist_to_target
    f_rep = np.zeros(2)
    f_tan = np.zeros(2)
    nearest_distance = repulsion_range
    nearest_direction = None

    if not hasattr(robot, 'escape_steps_remaining'):
        robot.escape_steps_remaining = 0
        robot.escape_tangent = np.zeros(2)
        robot.escape_normal = np.zeros(2)

    if robot.lidar is not None:
        for d, a in zip(robot.lidar.range_data, robot.lidar.angle_list):
            if d >= repulsion_range:
                continue

            actual_a = robot.state[2, 0] - np.pi / 2 + a
            beam_direction = np.array([np.cos(actual_a), np.sin(actual_a)])
            clearance = max(d, 0.25)
            weight = (repulsion_range - clearance) / repulsion_range
            f_rep -= beam_direction * (weight / (clearance ** 1.7))

            if d < nearest_distance:
                nearest_distance = d
                nearest_direction = beam_direction

    if nearest_direction is not None:
        tangent_left = np.array([-nearest_direction[1], nearest_direction[0]])
        tangent_right = -tangent_left
        tangent = tangent_left if np.dot(tangent_left, f_att) >= np.dot(tangent_right, f_att) else tangent_right
        tangent_weight = max(0.0, (slowdown_range - nearest_distance) / slowdown_range)
        f_tan = tangent * tangent_weight

        if nearest_distance < escape_distance:
            robot.escape_steps_remaining = escape_steps
            robot.escape_tangent = tangent
            robot.escape_normal = nearest_direction

    obstacle_factor = 0.0 if nearest_direction is None else max(
        0.0, (slowdown_range - nearest_distance) / slowdown_range
    )
    att_weight = max(0.25, 1.0 - 0.65 * obstacle_factor)
    speed_cap = cruise_speed * (1.0 - 0.55 * obstacle_factor)
    speed_cap = max(min_escape_speed, speed_cap)
    desired_speed = min(speed_cap, max(min_escape_speed * obstacle_factor, approach_gain * dist_to_target))

    if robot.escape_steps_remaining > 0:
        robot.escape_steps_remaining -= 1
        escape_speed = max(min_escape_speed * 1.8, desired_speed)
        escape_vel = (
            escape_turn_gain * escape_speed * robot.escape_tangent
            - backoff_gain * robot.escape_normal
        )

        if np.linalg.norm(escape_vel) > 1e-6:
            return escape_vel
    else:
        robot.escape_tangent = np.zeros(2)
        robot.escape_normal = np.zeros(2)

    vel = (
        desired_speed * att_weight * f_att
        + repulsion_gain * f_rep
        + tangential_gain * desired_speed * f_tan
    )

    return vel


target_points = generate_target_points(count=120)
target_colors = ['yellow'] * len(target_points)
target_plot = None


for robot in env.robot_list:
    robot.visited_points = set()
    robot.patrol_target = sample_patrol_point()

simulation_start_time = time.perf_counter()

for i in range(15000):
    elapsed_time = time.perf_counter() - simulation_start_time
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
        else:
            if np.linalg.norm(robot.patrol_target - pos) < 3.0:
                robot.patrol_target = sample_patrol_point()
            target = robot.patrol_target
        

        if target is not None:
            vel = compute_avoidance_velocity(robot, target, pos)
        else:

            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    rendezvous_logs = share_rendezvous_information(env.robot_list, env.components)
    for component, shared_count in rendezvous_logs:
        print(
            f"{elapsed_time:.1f}s rendezvous {component} "
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

    if all(any(p_idx in r.visited_points for r in env.robot_list) for p_idx in range(len(target_points))):
        elapsed_time = time.perf_counter() - simulation_start_time
        print(f"{elapsed_time:.1f} seconds: Mission Success! All target points visited.")
        break
