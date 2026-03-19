import numpy as np
from ir_sim.env import env_base
import matplotlib.pyplot as plt
from ir_sim.util.range_detection import range_seg_matrix, range_seg_seg


env = env_base('sim_kmeans.yaml', figsize=(19.2, 19.2))
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
rendezvous_cooldown = 3.0
min_new_shared_points_for_recluster = 2
hint_distance_weight = 0.75
opportunistic_switch_ratio = 0.55
opportunistic_capture_range = 6.0

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


def select_target_with_hints(robot, pos, target_points):
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
            return target_points[nearest_global_idx]

    assigned_set = set(assigned_candidates)
    best_idx = min(
        remaining_indices,
        key=lambda idx: (
            np.linalg.norm(target_points[idx] - pos) * hint_distance_weight
            if idx in assigned_set
            else np.linalg.norm(target_points[idx] - pos)
        )
    )
    return target_points[best_idx]


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
    robot.assigned_targets = []
    robot.last_rendezvous_group = tuple()
    robot.last_recluster_group = tuple()
    robot.last_recluster_time = -np.inf
    robot.last_shared_points_count = 0


for i in range(15000):
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
        if len(robot.visited_points) < len(target_points):
            target = select_target_with_hints(robot, pos, target_points)
        else:
            if np.linalg.norm(robot.patrol_target - pos) < 3.0:
                robot.patrol_target = sample_patrol_point()
            target = robot.patrol_target
        

        if target is not None:
            vel = compute_avoidance_velocity(robot, target, pos)
        else:

            vel = np.array([0.0, 0.0])
        
        vel_list.append(vel)

    rendezvous_events = share_rendezvous_information(env.robot_list, env.components)
    rendezvous_events = filter_rendezvous_events(
        rendezvous_events,
        env.robot_list,
        i * env.step_time,
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
            f"{i * env.step_time:.1f}s rendezvous {component} "
            f"shared {len(event['shared_points'])}/{len(target_points)} targets, "
            f"reclustered remaining {reassignment['remaining_count']} with K={reassignment['cluster_count']}"
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
