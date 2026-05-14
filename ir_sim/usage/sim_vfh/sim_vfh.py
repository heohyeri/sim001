import numpy as np
import time
from ir_sim.env import env_base
import matplotlib.pyplot as plt
from ir_sim.util.range_detection import range_seg_matrix, range_seg_seg
from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).with_name('sim_vfh.yaml')


def load_config(path):
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


config = load_config(CONFIG_PATH)
vfh_config = config.get('vfh_plus', {})

env = env_base('sim_vfh.yaml', figsize=(19.2, 19.2))
cruise_speed = vfh_config.get('cruise_speed', 7.0)
approach_gain = vfh_config.get('approach_gain', 1.5)
render_interval = 2
vfh_sector_count = vfh_config.get('sector_count', 72)
vfh_detection_range = vfh_config.get('detection_range', 8.0)
vfh_safety_distance = vfh_config.get('safety_distance', 0.8)
vfh_threshold = vfh_config.get('histogram_threshold', 0.35)
vfh_wide_valley_width = vfh_config.get('wide_valley_width', 12)
vfh_boundary_margin = vfh_config.get('boundary_margin', 3)
vfh_target_weight = vfh_config.get('target_weight', 5.0)
vfh_current_weight = vfh_config.get('current_weight', 2.0)
vfh_previous_weight = vfh_config.get('previous_weight', 2.0)
vfh_slowdown_heading_window = vfh_config.get('slowdown_heading_window', np.pi / 3)
vfh_max_heading_step = vfh_config.get('max_heading_step', 0.45)
diff_turn_time = vfh_config.get('diff_turn_time', 0.25)
wall_stop_margin = vfh_config.get('wall_stop_margin', 0.3)
use_omni_drive = vfh_config.get('use_omni_drive', False)
min_escape_speed = vfh_config.get('min_speed', 1.2)
rendezvous_cooldown = 3.0
min_new_shared_points_for_recluster = 2
hint_distance_weight = 0.75
opportunistic_switch_ratio = 0.55
opportunistic_capture_range = 6.0
stuck_window = vfh_config.get('stuck_window', 35)
stuck_min_travel = vfh_config.get('stuck_min_travel', 1.0)
stuck_min_target_progress = vfh_config.get('stuck_min_target_progress', 0.5)
target_block_duration = vfh_config.get('target_block_duration', 300)
oscillation_window = vfh_config.get('oscillation_window', 30)
oscillation_min_flips = vfh_config.get('oscillation_min_flips', 8)
oscillation_max_travel = vfh_config.get('oscillation_max_travel', 2.0)
oscillation_min_target_progress = vfh_config.get('oscillation_min_target_progress', 0.5)
immobility_max_travel = vfh_config.get('immobility_max_travel', 0.4)
immobility_max_avg_speed = vfh_config.get('immobility_max_avg_speed', 0.25)
immobility_min_target_progress = vfh_config.get('immobility_min_target_progress', 0.3)
world_width = config.get('world', {}).get('world_width', 180)
world_height = config.get('world', {}).get('world_height', 180)
world_offset_x = config.get('world', {}).get('offset_x', 0)
world_offset_y = config.get('world', {}).get('offset_y', 0)

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
    rendezvous_events = []

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

        component = sorted(component)
        component_changed = False
        for idx in component:
            if robot_list[idx].visited_points != shared_points:
                component_changed = True
            robot_list[idx].visited_points = shared_points.copy()

        component_tuple = tuple(component)
        group_changed = any(
            getattr(robot_list[idx], 'last_rendezvous_group', tuple()) != component_tuple
            for idx in component
        )

        for idx in component:
            robot_list[idx].last_rendezvous_group = component_tuple

        if len(component) > 1 and (component_changed or group_changed):
            rendezvous_events.append(
                {
                    'component': component,
                    'shared_points': shared_points.copy(),
                    'group_changed': group_changed,
                }
            )

    return rendezvous_events


def filter_rendezvous_events(events, robot_list, current_time, cooldown, min_new_shared_points):
    filtered_events = []

    for event in events:
        component_tuple = tuple(event['component'])
        last_time = max(
            getattr(robot_list[idx], 'last_recluster_time', -np.inf)
            for idx in event['component']
        )
        previous_shared_count = max(
            getattr(robot_list[idx], 'last_shared_points_count', 0)
            for idx in event['component']
        )
        new_shared_points = len(event['shared_points']) - previous_shared_count

        if current_time - last_time < cooldown:
            continue

        if new_shared_points < min_new_shared_points:
            continue

        for idx in event['component']:
            robot_list[idx].last_recluster_group = component_tuple
            robot_list[idx].last_recluster_time = current_time
            robot_list[idx].last_shared_points_count = len(event['shared_points'])

        filtered_events.append(event)

    return filtered_events


def initialize_centroids(points, k, rng):
    indices = rng.choice(len(points), size=k, replace=False)
    return points[indices].astype(float, copy=True)


def assign_points_to_centroids(points, centroids):
    distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def run_kmeans(points, k, max_iter=25, seed=0):
    if len(points) == 0:
        return np.array([], dtype=int), np.empty((0, 2))

    effective_k = min(max(1, k), len(points))
    rng = np.random.default_rng(seed)
    centroids = initialize_centroids(points, effective_k, rng)

    for _ in range(max_iter):
        labels = assign_points_to_centroids(points, centroids)
        new_centroids = centroids.copy()

        for cluster_idx in range(effective_k):
            cluster_points = points[labels == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = cluster_points.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    labels = assign_points_to_centroids(points, centroids)
    return labels, centroids


def order_target_indices(robot_position, target_indices, target_points):
    remaining = list(target_indices)
    ordered = []
    current = np.array(robot_position, dtype=float)

    while remaining:
        next_idx = min(
            remaining,
            key=lambda idx: np.linalg.norm(target_points[idx] - current)
        )
        ordered.append(next_idx)
        current = target_points[next_idx]
        remaining.remove(next_idx)

    return ordered


def assign_clusters_to_robots(component, centroids, robot_list):
    available_clusters = list(range(len(centroids)))
    assignments = {}

    for robot_idx in sorted(
        component,
        key=lambda idx: min(
            np.linalg.norm(np.squeeze(robot_list[idx].state[0:2]) - centroid)
            for centroid in centroids
        )
    ):
        if not available_clusters:
            assignments[robot_idx] = None
            continue

        robot_pos = np.squeeze(robot_list[robot_idx].state[0:2])
        chosen_cluster = min(
            available_clusters,
            key=lambda cluster_idx: np.linalg.norm(robot_pos - centroids[cluster_idx])
        )
        assignments[robot_idx] = chosen_cluster
        available_clusters.remove(chosen_cluster)

    return assignments


def reassign_targets_for_rendezvous(event, robot_list, target_points, iteration):
    component = event['component']
    shared_points = event['shared_points']
    remaining_indices = [
        idx for idx in range(len(target_points))
        if idx not in shared_points
    ]

    for robot_idx in component:
        robot = robot_list[robot_idx]
        robot.assigned_targets = [
            idx for idx in robot.assigned_targets
            if idx in remaining_indices and idx not in robot.visited_points
        ]

    if not remaining_indices:
        return {
            'component': component,
            'remaining_count': 0,
            'cluster_count': len(component),
        }

    remaining_points = target_points[remaining_indices]
    labels, centroids = run_kmeans(
        remaining_points,
        k=len(component),
        seed=iteration + sum(component) + len(shared_points)
    )

    robot_cluster_map = assign_clusters_to_robots(component, centroids, robot_list)

    for robot_idx in component:
        cluster_idx = robot_cluster_map[robot_idx]
        if cluster_idx is None:
            robot_list[robot_idx].assigned_targets = []
            continue

        robot_targets = [
            remaining_indices[point_offset]
            for point_offset, label in enumerate(labels)
            if label == cluster_idx
        ]
        robot_position = np.squeeze(robot_list[robot_idx].state[0:2])
        robot_list[robot_idx].assigned_targets = order_target_indices(
            robot_position,
            robot_targets,
            target_points
        )

    return {
        'component': component,
        'remaining_count': len(remaining_indices),
        'cluster_count': len(centroids),
    }


def select_target_with_hints(robot, pos, target_points, iteration):
    blocked_targets = getattr(robot, 'blocked_targets', {})
    robot.blocked_targets = {
        idx: unblock_step
        for idx, unblock_step in blocked_targets.items()
        if unblock_step > iteration and idx not in robot.visited_points
    }

    remaining_indices = [
        idx for idx in range(len(target_points))
        if idx not in robot.visited_points and idx not in robot.blocked_targets
    ]

    if not remaining_indices:
        remaining_indices = [
            idx for idx in range(len(target_points))
            if idx not in robot.visited_points
        ]

    if not remaining_indices:
        return None

    nearest_global_idx = min(
        remaining_indices,
        key=lambda idx: np.linalg.norm(target_points[idx] - pos)
    )
    nearest_global_dist = np.linalg.norm(target_points[nearest_global_idx] - pos)

    assigned_candidates = [
        idx for idx in robot.assigned_targets
        if idx in remaining_indices
    ]

    if assigned_candidates:
        nearest_assigned_idx = min(
            assigned_candidates,
            key=lambda idx: np.linalg.norm(target_points[idx] - pos)
        )
        nearest_assigned_dist = np.linalg.norm(target_points[nearest_assigned_idx] - pos)

        # Treat cluster output as a preference, not a hard constraint.
        if (
            nearest_global_idx != nearest_assigned_idx
            and nearest_global_dist <= opportunistic_capture_range
            and nearest_global_dist < opportunistic_switch_ratio * nearest_assigned_dist
        ):
            return nearest_global_idx

    assigned_set = set(assigned_candidates)
    best_idx = min(
        remaining_indices,
        key=lambda idx: (
            np.linalg.norm(target_points[idx] - pos) * hint_distance_weight
            if idx in assigned_set
            else np.linalg.norm(target_points[idx] - pos)
        )
    )
    return best_idx


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


def update_stuck_history(robot, pos, target):
    target_distance = np.linalg.norm(target - pos)
    history = getattr(robot, 'stuck_history', [])
    history.append((pos.copy(), target_distance))

    if len(history) > stuck_window:
        history = history[-stuck_window:]

    robot.stuck_history = history


def robot_is_stuck(robot):
    history = getattr(robot, 'stuck_history', [])

    if len(history) < stuck_window:
        return False

    start_pos, start_target_distance = history[0]
    end_pos, end_target_distance = history[-1]
    traveled = np.linalg.norm(end_pos - start_pos)
    target_progress = start_target_distance - end_target_distance

    return traveled < stuck_min_travel and target_progress < stuck_min_target_progress


def update_oscillation_history(robot, pos, target, vel):
    target_offset = target - pos
    target_distance = np.linalg.norm(target_offset)
    speed = np.linalg.norm(vel)
    side = 0

    if target_distance > 1e-6 and speed > 1e-6:
        target_angle = np.arctan2(target_offset[1], target_offset[0])
        move_angle = np.arctan2(vel[1], vel[0])
        angle_error = to_pi(move_angle - target_angle)
        if abs(angle_error) > 0.1:
            side = 1 if angle_error > 0 else -1

    history = getattr(robot, 'oscillation_history', [])
    history.append((pos.copy(), target_distance, side, speed))

    if len(history) > oscillation_window:
        history = history[-oscillation_window:]

    robot.oscillation_history = history


def robot_is_oscillating(robot):
    history = getattr(robot, 'oscillation_history', [])

    if len(history) < oscillation_window:
        return False

    start_pos, start_target_distance, _, _ = history[0]
    end_pos, end_target_distance, _, _ = history[-1]
    traveled = np.linalg.norm(end_pos - start_pos)
    target_progress = start_target_distance - end_target_distance
    sides = [side for _, _, side, _ in history if side != 0]

    if len(sides) < 2:
        return False

    flips = sum(1 for prev, cur in zip(sides, sides[1:]) if prev != cur)
    return (
        flips >= oscillation_min_flips
        and traveled < oscillation_max_travel
        and target_progress < oscillation_min_target_progress
    )


def robot_is_immobile(robot):
    history = getattr(robot, 'oscillation_history', [])

    if len(history) < oscillation_window:
        return False

    start_pos, start_target_distance, _, _ = history[0]
    end_pos, end_target_distance, _, _ = history[-1]
    traveled = np.linalg.norm(end_pos - start_pos)
    target_progress = start_target_distance - end_target_distance
    avg_speed = np.mean([speed for _, _, _, speed in history])

    return (
        traveled < immobility_max_travel
        and avg_speed < immobility_max_avg_speed
        and target_progress < immobility_min_target_progress
    )


def angle_to_sector(angle):
    sector_angle = 2 * np.pi / vfh_sector_count
    return int(np.floor((to_pi(angle) + np.pi) / sector_angle)) % vfh_sector_count


def sector_to_angle(sector):
    sector_angle = 2 * np.pi / vfh_sector_count
    return to_pi(-np.pi + (sector + 0.5) * sector_angle)


def circular_sector_distance(first, second):
    direct = abs(first - second)
    return min(direct, vfh_sector_count - direct)


def clamp_sector_to_valley(target_sector, start, end, width):
    if width <= 0:
        return start

    sectors = [(start + offset) % vfh_sector_count for offset in range(width)]
    return min(sectors, key=lambda sector: circular_sector_distance(sector, target_sector))


def find_open_valleys(blocked):
    if np.all(blocked):
        return []

    if not np.any(blocked):
        return [(0, vfh_sector_count - 1, vfh_sector_count)]

    start_scan = (int(np.flatnonzero(blocked)[0]) + 1) % vfh_sector_count
    valleys = []
    current_width = 0
    current_start = None

    for offset in range(vfh_sector_count):
        sector = (start_scan + offset) % vfh_sector_count

        if blocked[sector]:
            if current_width > 0:
                valleys.append((current_start, (sector - 1) % vfh_sector_count, current_width))
                current_width = 0
                current_start = None
            continue

        if current_width == 0:
            current_start = sector
        current_width += 1

    if current_width > 0:
        end = (start_scan + vfh_sector_count - 1) % vfh_sector_count
        valleys.append((current_start, end, current_width))

    return valleys


def build_vfh_histogram(robot):
    histogram = np.zeros(vfh_sector_count)
    nearest_distance = vfh_detection_range

    if robot.lidar is None:
        return histogram, nearest_distance

    robot_radius = getattr(robot, 'radius', 0.2)
    inflation_radius = robot_radius + vfh_safety_distance
    sector_angle = 2 * np.pi / vfh_sector_count

    for distance, lidar_angle in zip(robot.lidar.range_data, robot.lidar.angle_list):
        if not np.isfinite(distance) or distance >= vfh_detection_range:
            continue

        clearance = max(distance, 0.05)
        obstacle_angle = robot.state[2, 0] - np.pi / 2 + lidar_angle
        center_sector = angle_to_sector(obstacle_angle)
        nearest_distance = min(nearest_distance, clearance)

        expansion_angle = np.arcsin(min(1.0, inflation_radius / clearance))
        expansion_sectors = int(np.ceil(expansion_angle / sector_angle))
        obstacle_weight = ((vfh_detection_range - clearance) / vfh_detection_range) ** 2

        for offset in range(-expansion_sectors, expansion_sectors + 1):
            sector = (center_sector + offset) % vfh_sector_count
            taper = 1.0 - 0.35 * abs(offset) / max(1, expansion_sectors)
            histogram[sector] = max(histogram[sector], obstacle_weight * taper)

    return histogram, nearest_distance


def candidate_sectors_from_valley(valley, target_sector):
    start, end, width = valley

    if width >= vfh_wide_valley_width:
        left_candidate = (start + min(vfh_boundary_margin, width - 1)) % vfh_sector_count
        right_candidate = (end - min(vfh_boundary_margin, width - 1)) % vfh_sector_count
        target_candidate = clamp_sector_to_valley(target_sector, start, end, width)
        return {left_candidate, right_candidate, target_candidate}

    return {clamp_sector_to_valley(target_sector, start, end, width)}


def select_vfh_heading(robot, target_angle, current_heading, blocked, histogram):
    target_sector = angle_to_sector(target_angle)
    valleys = find_open_valleys(blocked)

    candidates = []
    for valley in valleys:
        candidates.extend(candidate_sectors_from_valley(valley, target_sector))

    if not candidates:
        candidates = range(vfh_sector_count)

    previous_heading = getattr(robot, 'previous_vfh_heading', current_heading)

    def heading_cost(sector):
        heading = sector_to_angle(sector)
        return (
            vfh_target_weight * abs(to_pi(heading - target_angle))
            + vfh_current_weight * abs(to_pi(heading - current_heading))
            + vfh_previous_weight * abs(to_pi(heading - previous_heading))
            + histogram[sector]
        )

    best_sector = min(candidates, key=heading_cost)
    return sector_to_angle(best_sector)


def limit_heading_change(previous_heading, desired_heading):
    delta = to_pi(desired_heading - previous_heading)
    limited_delta = np.clip(delta, -vfh_max_heading_step, vfh_max_heading_step)
    return to_pi(previous_heading + limited_delta)


def clearance_along_heading(robot, heading):
    if robot.lidar is None:
        return vfh_detection_range

    nearest_clearance = vfh_detection_range
    half_window = vfh_slowdown_heading_window / 2

    for distance, lidar_angle in zip(robot.lidar.range_data, robot.lidar.angle_list):
        if not np.isfinite(distance) or distance >= vfh_detection_range:
            continue

        ray_heading = robot.state[2, 0] - np.pi / 2 + lidar_angle
        if abs(to_pi(ray_heading - heading)) <= half_window:
            nearest_clearance = min(nearest_clearance, max(distance, 0.05))

    return nearest_clearance


def safe_linear_speed(robot, linear_speed, heading, angular_speed=0.0):
    if linear_speed <= 0:
        return 0.0

    pos = np.squeeze(robot.state[0:2])
    step_time = getattr(robot, 'step_time', env.step_time)
    mid_heading = heading + 0.5 * angular_speed * step_time
    direction = np.array([np.cos(mid_heading), np.sin(mid_heading)])
    world_min = np.array([world_offset_x, world_offset_y]) + getattr(robot, 'radius', 0.0) + wall_stop_margin
    world_max = np.array([world_offset_x + world_width, world_offset_y + world_height]) - getattr(robot, 'radius', 0.0) - wall_stop_margin
    max_boundary_travel = np.inf

    for axis in range(2):
        if direction[axis] > 1e-9:
            max_boundary_travel = min(max_boundary_travel, (world_max[axis] - pos[axis]) / direction[axis])
        elif direction[axis] < -1e-9:
            max_boundary_travel = min(max_boundary_travel, (world_min[axis] - pos[axis]) / direction[axis])

    if max_boundary_travel <= 0:
        return 0.0

    if np.isfinite(max_boundary_travel):
        linear_speed = min(linear_speed, max_boundary_travel / step_time)

    travel = linear_speed * step_time + getattr(robot, 'radius', 0.0) + wall_stop_margin
    next_pos = pos + travel * direction
    motion_segment = [pos, next_pos]
    nearest_hit_range = None

    blocked_by_map, _, map_range = range_seg_matrix(
        motion_segment,
        env.components['map_matrix'],
        env.components['xy_reso'],
        robot.lidar.point_step_weight if robot.lidar is not None else 2,
        env.components['offset']
    )
    if blocked_by_map:
        nearest_hit_range = map_range

    for line in env.components['obs_lines'].obs_line_states:
        wall_segment = [
            np.array([line[0], line[1]], dtype=float),
            np.array([line[2], line[3]], dtype=float)
        ]
        hit_wall, _, wall_range = range_seg_seg(motion_segment, wall_segment)
        if hit_wall:
            if nearest_hit_range is None:
                nearest_hit_range = wall_range
            else:
                nearest_hit_range = min(nearest_hit_range, wall_range)

    if nearest_hit_range is None:
        return linear_speed

    clearance = nearest_hit_range - getattr(robot, 'radius', 0.0) - wall_stop_margin
    return max(0.0, min(linear_speed, clearance / step_time))



def compute_avoidance_command(robot, target, pos):
    target_offset = target - pos
    dist_to_target = np.linalg.norm(target_offset)

    if dist_to_target < 1e-6:
        return np.zeros(2)

    target_angle = np.arctan2(target_offset[1], target_offset[0])
    current_heading = getattr(robot, 'previous_vfh_heading', robot.state[2, 0])
    histogram, _ = build_vfh_histogram(robot)
    blocked = histogram > vfh_threshold
    raw_heading = select_vfh_heading(robot, target_angle, current_heading, blocked, histogram)
    selected_heading = limit_heading_change(current_heading, raw_heading)
    robot.previous_vfh_heading = selected_heading

    max_drive_speed = robot.vel_max[0, 0]
    desired_speed = min(max_drive_speed, max(min_escape_speed, approach_gain * dist_to_target))

    if robot.mode == 'diff' and not use_omni_drive:
        heading_error = to_pi(selected_heading - robot.state[2, 0])
        w_max = robot.vel_max[1, 0]
        angular_speed = np.clip(heading_error / diff_turn_time, -w_max, w_max)
        linear_speed = safe_linear_speed(robot, desired_speed, robot.state[2, 0], angular_speed)

        return np.array([linear_speed, angular_speed])

    linear_speed = safe_linear_speed(robot, desired_speed, selected_heading)
    return linear_speed * np.array([np.cos(selected_heading), np.sin(selected_heading)])


def robot_step_omni_drive(robot_list, vel_list):
    for robot, vel in zip(robot_list, vel_list):
        if isinstance(vel, list):
            vel = np.array(vel, ndmin=2).T
        if vel.shape == (2,):
            vel = vel[:, np.newaxis]

        speed = np.linalg.norm(vel)
        speed_limit = robot.vel_max[0, 0]
        if speed > speed_limit:
            vel = vel / speed * speed_limit
            speed = speed_limit

        robot.previous_state = robot.state.copy()
        next_pos = robot.state[0:2] + vel * robot.step_time
        world_min = np.array([[world_offset_x], [world_offset_y]]) + robot.radius
        world_max = np.array([[world_offset_x + world_width], [world_offset_y + world_height]]) - robot.radius
        robot.state[0:2] = np.clip(next_pos, world_min, world_max)

        if speed > 1e-6:
            robot.state[2, 0] = to_pi(np.arctan2(vel[1, 0], vel[0, 0]))

        robot.vel_omni = vel
        robot.vel_diff = np.array([[speed], [0.0]])
        robot.arrive()


target_points = generate_target_points(count=20)
target_colors = ['yellow'] * len(target_points)
target_plot = None


for robot in env.robot_list:
    robot.visited_points = set()
    robot.patrol_target = sample_patrol_point()
    robot.assigned_targets = []
    robot.last_rendezvous_group = tuple()
    robot.last_recluster_group = tuple()
    robot.last_recluster_time = -np.inf
    robot.last_shared_points_count = 0
    robot.blocked_targets = {}
    robot.stuck_history = []
    robot.oscillation_history = []

simulation_start_time = time.perf_counter()

for i in range(15000):
    simulation_time = i * env.step_time
    elapsed_time = time.perf_counter() - simulation_start_time
    vel_list = []
    
    for r_idx, robot in enumerate(env.robot_list):

        robot.cal_lidar_range(env.components)
        pos = np.squeeze(robot.state[0:2])
        

        for p_idx, pt in enumerate(target_points):
            if np.linalg.norm(pos - pt) < 1.2:
                robot.visited_points.add(p_idx)

        robot.assigned_targets = [
            idx for idx in robot.assigned_targets
            if idx not in robot.visited_points
        ]

        target = None
        target_idx = None
        if len(robot.visited_points) < len(target_points):
            target_idx = select_target_with_hints(robot, pos, target_points, i)
            if target_idx is not None:
                target = target_points[target_idx]
        else:
            if np.linalg.norm(robot.patrol_target - pos) < 3.0:
                robot.patrol_target = sample_patrol_point()
            target = robot.patrol_target

        if target is not None and target_idx is not None:
            update_stuck_history(robot, pos, target)
            if robot_is_stuck(robot):
                robot.blocked_targets[target_idx] = i + target_block_duration
                robot.stuck_history = []
                robot.oscillation_history = []
                target_idx = select_target_with_hints(robot, pos, target_points, i)
                target = target_points[target_idx] if target_idx is not None else None
        

        if target is not None:
            vel = compute_avoidance_command(robot, target, pos)
            if target_idx is not None:
                update_oscillation_history(robot, pos, target, vel)
                if robot_is_oscillating(robot) or robot_is_immobile(robot):
                    robot.blocked_targets[target_idx] = i + target_block_duration
                    robot.stuck_history = []
                    robot.oscillation_history = []
                    target_idx = select_target_with_hints(robot, pos, target_points, i)
                    target = target_points[target_idx] if target_idx is not None else None
                    vel = compute_avoidance_command(robot, target, pos) if target is not None else np.array([0.0, 0.0])
        else:

            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    rendezvous_events = share_rendezvous_information(env.robot_list, env.components)
    rendezvous_events = filter_rendezvous_events(
        rendezvous_events,
        env.robot_list,
        simulation_time,
        rendezvous_cooldown,
        min_new_shared_points_for_recluster
    )
    for event in rendezvous_events:
        reassignment = reassign_targets_for_rendezvous(
            event,
            env.robot_list,
            target_points,
            i
        )
        component = reassignment['component']
        print(
            f"{elapsed_time:.1f}s rendezvous {component} "
            f"shared {len(event['shared_points'])}/{len(target_points)} targets, "
            f"reclustered remaining {reassignment['remaining_count']} with K={reassignment['cluster_count']}"
        )

    if use_omni_drive:
        robot_step_omni_drive(env.robot_list, vel_list)
    else:
        env.robot_step(vel_list, vel_type='diff', stop=False)
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
