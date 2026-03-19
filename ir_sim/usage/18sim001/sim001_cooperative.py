import heapq
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from ir_sim.env import env_base
from ir_sim.util.range_detection import range_seg_matrix, range_seg_seg


WORLD_SIZE = (100, 100)
TARGET_COUNT = 50
TARGET_CAPTURE_RADIUS = 1.2
CRUISE_SPEED = 10.0
APPROACH_GAIN = 1.5

REPULSION_RANGE = 6.0
REPULSION_GAIN = 0.6

SAFE_CLEARANCE = 5.0
ANGLE_WINDOW = np.deg2rad(20)
CLEARANCE_WEIGHT = 0.35
SPEED_CLEARANCE_GAIN = 1.5
SPEED_CLEARANCE_OFFSET = 0.8

DETOUR_LOOKAHEAD = 10.0
DETOUR_MARGIN = 0.8
DETOUR_HOLD_STEPS = 20
TARGET_PROGRESS_EPS = 0.2
TARGET_STALL_THRESHOLD = 5

GRID_RESOLUTION = 1.0
OBSTACLE_INFLATION = 1.6
WAYPOINT_REACHED_RADIUS = 1.5
PATH_STALL_THRESHOLD = 10
ASTAR_DIAGONAL = True

ESCAPE_SPEED = 6.0
ESCAPE_STEPS = 10
STUCK_DISTANCE = 0.15
STUCK_STEP_THRESHOLD = 8

CENTRALIZED_INIT = True
REBALANCE_COOLDOWN = 12

RENDER_INTERVAL = 2
MAX_STEPS = 15000
TARGET_SEED = 7


env = env_base("sim001.yaml", figsize=(19.2, 19.2))
rng = np.random.default_rng(TARGET_SEED)


def robot_position(robot):
    return np.squeeze(robot.state[0:2]).astype(float)


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
    world_size=WORLD_SIZE,
    margin=8.0,
    min_spacing=10.0,
    wall_clearance=4.0,
    seed=None,
):
    local_rng = np.random.default_rng(seed)
    width, height = world_size
    points = []
    wall_segments = env.components["obs_lines"].obs_line_states

    while len(points) < count:
        candidate = np.array(
            [
                local_rng.uniform(margin, width - margin),
                local_rng.uniform(margin, height - margin),
            ]
        )

        if any(np.linalg.norm(candidate - existing) < min_spacing for existing in points):
            continue

        if any(
            point_to_segment_distance(candidate, segment) < wall_clearance
            for segment in wall_segments
        ):
            continue

        points.append(candidate)

    return np.array(points)


def to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def robots_in_contact(observer, target, components):
    # This is line-of-sight communication contact, not strict physical collision.
    if observer.lidar is None:
        return False

    observer_pos = robot_position(observer)
    target_pos = robot_position(target)
    relative = target_pos - observer_pos
    distance = np.linalg.norm(relative)

    if distance == 0 or distance > observer.lidar.range_max:
        return False

    heading = observer.state[2, 0] - np.pi / 2
    bearing = to_pi(np.arctan2(relative[1], relative[0]) - heading)

    if not (observer.lidar.angle_min <= bearing <= observer.lidar.angle_max):
        return False

    segment = [observer_pos, target_pos]
    blocked_by_map, _, map_range = range_seg_matrix(
        segment,
        components["map_matrix"],
        components["xy_reso"],
        observer.lidar.point_step_weight,
        components["offset"],
    )

    if blocked_by_map and map_range < distance:
        return False

    for line in components["obs_lines"].obs_line_states:
        wall_segment = [
            np.array([line[0], line[1]], dtype=float),
            np.array([line[2], line[3]], dtype=float),
        ]
        blocked_by_wall, _, wall_range = range_seg_seg(segment, wall_segment)
        if blocked_by_wall and wall_range < distance:
            return False

    return True


def build_contact_components(robot_list, components):
    robot_count = len(robot_list)
    adjacency = {idx: set() for idx in range(robot_count)}

    for idx in range(robot_count):
        for next_idx in range(idx + 1, robot_count):
            can_share = robots_in_contact(
                robot_list[idx], robot_list[next_idx], components
            ) or robots_in_contact(robot_list[next_idx], robot_list[idx], components)

            if can_share:
                adjacency[idx].add(next_idx)
                adjacency[next_idx].add(idx)

    visited = [False] * robot_count
    components_list = []

    for start_idx in range(robot_count):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        component = []

        while stack:
            idx = stack.pop()
            if visited[idx]:
                continue

            visited[idx] = True
            component.append(idx)

            for neighbor in adjacency[idx]:
                if not visited[neighbor]:
                    stack.append(neighbor)

        components_list.append(sorted(component))

    return components_list


def initialize_kmeans(points, cluster_count):
    centroids = [points[rng.integers(len(points))]]

    while len(centroids) < cluster_count:
        centroid_array = np.array(centroids)
        distances = np.linalg.norm(
            points[:, np.newaxis, :] - centroid_array[np.newaxis, :, :], axis=2
        )
        farthest_idx = np.argmax(np.min(distances, axis=1))
        centroids.append(points[farthest_idx])

    return np.array(centroids)


def kmeans_partition(points, cluster_count, max_iters=25):
    if len(points) == 0 or cluster_count == 0:
        return np.array([], dtype=int), np.empty((0, 2))

    if len(points) <= cluster_count:
        labels = np.arange(len(points), dtype=int)
        return labels, points.copy()

    centroids = initialize_kmeans(points, cluster_count)

    for _ in range(max_iters):
        distances = np.linalg.norm(
            points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for cluster_idx in range(cluster_count):
            members = points[labels == cluster_idx]
            if len(members) == 0:
                farthest_idx = np.argmax(np.min(distances, axis=1))
                new_centroids[cluster_idx] = points[farthest_idx]
            else:
                new_centroids[cluster_idx] = members.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return labels, centroids


def lidar_world_angles(robot):
    return robot.state[2, 0] - np.pi / 2 + robot.lidar.angle_list


def angle_distance(angle_list, target_angle):
    return np.abs((angle_list - target_angle + np.pi) % (2 * np.pi) - np.pi)


def clearance_for_direction(robot, world_angle):
    beam_angles = lidar_world_angles(robot)
    angular_distance = angle_distance(beam_angles, world_angle)
    mask = angular_distance <= ANGLE_WINDOW

    if np.any(mask):
        return float(np.min(robot.lidar.range_data[mask]))

    nearest_idx = int(np.argmin(angular_distance))
    return float(robot.lidar.range_data[nearest_idx])


def grid_shape():
    return (
        int(np.ceil(WORLD_SIZE[0] / GRID_RESOLUTION)),
        int(np.ceil(WORLD_SIZE[1] / GRID_RESOLUTION)),
    )


def world_to_grid(point):
    width_cells, height_cells = grid_shape()
    gx = int(np.clip(np.floor(point[0] / GRID_RESOLUTION), 0, width_cells - 1))
    gy = int(np.clip(np.floor(point[1] / GRID_RESOLUTION), 0, height_cells - 1))
    return (gx, gy)


def grid_to_world(cell):
    return np.array(
        [
            (cell[0] + 0.5) * GRID_RESOLUTION,
            (cell[1] + 0.5) * GRID_RESOLUTION,
        ]
    )


def build_occupancy_grid():
    width_cells, height_cells = grid_shape()
    occupancy = np.zeros((width_cells, height_cells), dtype=bool)
    line_segments = env.components["obs_lines"].obs_line_states
    inflation = OBSTACLE_INFLATION

    for gx in range(width_cells):
        for gy in range(height_cells):
            point = grid_to_world((gx, gy))

            if point[0] < inflation or point[0] > WORLD_SIZE[0] - inflation:
                occupancy[gx, gy] = True
                continue

            if point[1] < inflation or point[1] > WORLD_SIZE[1] - inflation:
                occupancy[gx, gy] = True
                continue

            for segment in line_segments:
                if point_to_segment_distance(point, segment) <= inflation:
                    occupancy[gx, gy] = True
                    break

    return occupancy


def in_bounds(cell, occupancy):
    return 0 <= cell[0] < occupancy.shape[0] and 0 <= cell[1] < occupancy.shape[1]


def nearest_free_cell(start_cell, occupancy, max_radius=8):
    if in_bounds(start_cell, occupancy) and not occupancy[start_cell]:
        return start_cell

    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = (start_cell[0] + dx, start_cell[1] + dy)
                if in_bounds(candidate, occupancy) and not occupancy[candidate]:
                    return candidate

    return None


def grid_neighbors(cell):
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if ASTAR_DIAGONAL:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in offsets:
        next_cell = (cell[0] + dx, cell[1] + dy)
        step_cost = np.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
        yield next_cell, step_cost


def heuristic(cell, goal_cell):
    return np.linalg.norm(np.array(cell) - np.array(goal_cell))


def astar_search(start_cell, goal_cell, occupancy):
    open_heap = [(heuristic(start_cell, goal_cell), 0.0, start_cell)]
    parent = {start_cell: None}
    g_cost = {start_cell: 0.0}

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current == goal_cell:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        if current_cost > g_cost[current] + 1e-9:
            continue

        for next_cell, step_cost in grid_neighbors(current):
            if not in_bounds(next_cell, occupancy) or occupancy[next_cell]:
                continue

            new_cost = current_cost + step_cost
            if new_cost < g_cost.get(next_cell, float("inf")):
                g_cost[next_cell] = new_cost
                parent[next_cell] = current
                priority = new_cost + heuristic(next_cell, goal_cell)
                heapq.heappush(open_heap, (priority, new_cost, next_cell))

    return []


def line_of_sight_cells(start_cell, end_cell, occupancy):
    start = grid_to_world(start_cell)
    end = grid_to_world(end_cell)
    distance = np.linalg.norm(end - start)
    sample_count = max(2, int(np.ceil(distance / (GRID_RESOLUTION * 0.5))))

    for idx in range(1, sample_count):
        alpha = idx / sample_count
        point = start + alpha * (end - start)
        cell = world_to_grid(point)
        if occupancy[cell]:
            return False

    return True


def simplify_path_cells(path_cells, occupancy):
    if not path_cells:
        return []

    simplified = [path_cells[0]]
    anchor_idx = 0

    while anchor_idx < len(path_cells) - 1:
        farthest_idx = anchor_idx + 1
        for next_idx in range(anchor_idx + 1, len(path_cells)):
            if line_of_sight_cells(path_cells[anchor_idx], path_cells[next_idx], occupancy):
                farthest_idx = next_idx
            else:
                break

        simplified.append(path_cells[farthest_idx])
        anchor_idx = farthest_idx

    return simplified


def reset_detour(robot):
    robot.detour_waypoint = None
    robot.detour_target_id = None
    robot.detour_hold_steps = 0


def reset_path(robot):
    robot.global_path = []
    robot.path_waypoint_idx = 0
    robot.path_target_id = None
    robot.path_goal_cell = None
    robot.path_stall_steps = 0
    robot.last_waypoint_distance = None


def invalidate_current_target_if_needed(robot):
    if robot.current_target is None:
        return

    invalid_assigned_target = (
        robot.use_assigned_targets and robot.current_target not in robot.assigned_targets
    )

    if robot.current_target in robot.visited_points or invalid_assigned_target:
        robot.current_target = None
        robot.last_target_distance = None
        robot.last_target_id = None
        robot.target_stall_steps = 0
        reset_detour(robot)
        reset_path(robot)


def snapshot_targets(targets):
    return tuple(sorted(targets))


def set_component_rebalance_memory(component, shared_points, target_pool, step):
    component_signature = tuple(component)
    shared_signature = snapshot_targets(shared_points)
    pool_signature = snapshot_targets(target_pool)

    for robot_idx in component:
        robot = env.robot_list[robot_idx]
        robot.last_component = component_signature
        robot.last_shared_signature = shared_signature
        robot.last_target_pool = pool_signature
        robot.last_rebalance_step = step


def allocate_component_targets(component, target_pool, target_points):
    if not target_pool:
        assignments = {robot_idx: [] for robot_idx in component}
    else:
        target_indices = sorted(target_pool)
        pool_points = target_points[target_indices]
        cluster_count = min(len(component), len(target_indices))
        labels, centroids = kmeans_partition(pool_points, cluster_count)

        clusters = []
        for cluster_idx in range(cluster_count):
            cluster_targets = [
                target_indices[target_pos]
                for target_pos, label in enumerate(labels)
                if label == cluster_idx
            ]
            clusters.append(sorted(cluster_targets))

        robot_positions = np.array([robot_position(env.robot_list[idx]) for idx in component])
        cost_matrix = np.linalg.norm(
            robot_positions[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        assignments = {robot_idx: [] for robot_idx in component}
        for row, col in zip(row_idx, col_idx):
            assignments[component[row]] = clusters[col]

    for robot_idx in component:
        robot = env.robot_list[robot_idx]
        robot.assigned_targets = set(assignments[robot_idx])
        robot.use_assigned_targets = len(component) > 1
        invalidate_current_target_if_needed(robot)

    return assignments


def initialize_target_assignments(robot_list, target_points, centralized_init=True):
    all_targets = set(range(len(target_points)))

    if centralized_init:
        allocate_component_targets(list(range(len(robot_list))), all_targets, target_points)
    else:
        for robot in robot_list:
            robot.assigned_targets = set()
            robot.use_assigned_targets = False
            invalidate_current_target_if_needed(robot)

    for idx, robot in enumerate(robot_list):
        set_component_rebalance_memory(
            [idx], robot.visited_points, robot.assigned_targets - robot.visited_points, 0
        )


def setup_robot_memory(robot_list):
    for idx, robot in enumerate(robot_list):
        robot.visited_points = set()
        robot.assigned_targets = set()
        robot.use_assigned_targets = False
        robot.current_target = None

        robot.last_component = (idx,)
        robot.last_shared_signature = tuple()
        robot.last_target_pool = tuple()
        robot.last_rebalance_step = -REBALANCE_COOLDOWN

        robot.last_position = robot_position(robot).copy()
        robot.stuck_steps = 0
        robot.escape_steps = 0
        robot.escape_direction = np.zeros(2)

        robot.last_target_distance = None
        robot.last_target_id = None
        robot.target_stall_steps = 0
        robot.detour_waypoint = None
        robot.detour_target_id = None
        robot.detour_hold_steps = 0

        robot.global_path = []
        robot.path_waypoint_idx = 0
        robot.path_target_id = None
        robot.path_goal_cell = None
        robot.path_stall_steps = 0
        robot.last_waypoint_distance = None


def rebalance_components(components, target_points, step):
    logs = []
    all_target_indices = set(range(len(target_points)))

    for component in components:
        shared_points = set()
        for robot_idx in component:
            shared_points |= env.robot_list[robot_idx].visited_points

        for robot_idx in component:
            env.robot_list[robot_idx].visited_points = shared_points.copy()

        target_pool = set()
        for robot_idx in component:
            target_pool |= env.robot_list[robot_idx].assigned_targets
        target_pool -= shared_points

        if len(component) > 1 and not target_pool:
            if any(not env.robot_list[robot_idx].use_assigned_targets for robot_idx in component):
                target_pool = all_target_indices - shared_points

        component_signature = tuple(component)
        shared_signature = snapshot_targets(shared_points)
        target_pool_signature = snapshot_targets(target_pool)

        state_changed = any(
            env.robot_list[robot_idx].last_component != component_signature
            or env.robot_list[robot_idx].last_shared_signature != shared_signature
            or env.robot_list[robot_idx].last_target_pool != target_pool_signature
            for robot_idx in component
        )

        cooldown_ready = all(
            step - env.robot_list[robot_idx].last_rebalance_step >= REBALANCE_COOLDOWN
            for robot_idx in component
        )

        should_rebalance = len(component) > 1 and target_pool and state_changed and cooldown_ready

        if should_rebalance:
            assignments = allocate_component_targets(component, target_pool, target_points)
            assignment_sizes = {
                robot_idx: len(assigned) for robot_idx, assigned in assignments.items()
            }
            logs.append((component_signature, len(shared_points), assignment_sizes))
            set_component_rebalance_memory(component, shared_points, target_pool, step)
            continue

        if len(component) == 1 or not target_pool or not state_changed:
            set_component_rebalance_memory(component, shared_points, target_pool, step)

    return logs


def choose_target(robot, target_points):
    if robot.use_assigned_targets:
        candidate_targets = robot.assigned_targets - robot.visited_points
    else:
        candidate_targets = set(range(len(target_points))) - robot.visited_points

    if robot.current_target in candidate_targets:
        return target_points[robot.current_target]

    invalidate_current_target_if_needed(robot)

    if not candidate_targets:
        robot.current_target = None
        return None

    pos = robot_position(robot)
    candidate_targets = sorted(candidate_targets)
    distances = [np.linalg.norm(target_points[idx] - pos) for idx in candidate_targets]
    nearest_idx = candidate_targets[int(np.argmin(distances))]
    robot.current_target = nearest_idx
    robot.last_target_id = None
    reset_path(robot)
    return target_points[nearest_idx]


def select_safe_direction(robot, desired_direction):
    if robot.lidar is None:
        return desired_direction, np.inf

    desired_angle = np.arctan2(desired_direction[1], desired_direction[0])
    candidate_angles = np.concatenate((lidar_world_angles(robot), np.array([desired_angle])))

    best_safe_direction = None
    best_safe_score = -np.inf
    best_safe_clearance = 0.0

    best_fallback_direction = desired_direction
    best_fallback_score = -np.inf
    best_fallback_clearance = 0.0

    for candidate_angle in candidate_angles:
        direction = np.array([np.cos(candidate_angle), np.sin(candidate_angle)])
        clearance = clearance_for_direction(robot, candidate_angle)
        alignment = float(np.dot(direction, desired_direction))
        clearance_score = min(clearance / max(REPULSION_RANGE, 1e-6), 1.0)
        score = alignment + CLEARANCE_WEIGHT * clearance_score

        if score > best_fallback_score:
            best_fallback_score = score
            best_fallback_direction = direction
            best_fallback_clearance = clearance

        if clearance >= SAFE_CLEARANCE and score > best_safe_score:
            best_safe_score = score
            best_safe_direction = direction
            best_safe_clearance = clearance

    if best_safe_direction is not None:
        return best_safe_direction, best_safe_clearance

    return best_fallback_direction, best_fallback_clearance


def plan_detour_waypoint(robot, target):
    if robot.lidar is None:
        return None

    pos = robot_position(robot)
    current_distance = np.linalg.norm(target - pos)
    if current_distance < 1e-9:
        return None

    target_direction = (target - pos) / current_distance
    best_waypoint = None
    best_score = -np.inf

    for angle, clearance in zip(lidar_world_angles(robot), robot.lidar.range_data):
        usable_clearance = clearance - DETOUR_MARGIN
        travel_distance = min(DETOUR_LOOKAHEAD, usable_clearance)
        if travel_distance <= 1.0:
            continue

        direction = np.array([np.cos(angle), np.sin(angle)])
        waypoint = pos + travel_distance * direction
        projected_distance = np.linalg.norm(target - waypoint)
        progress = current_distance - projected_distance
        alignment = float(np.dot(direction, target_direction))
        score = 3.0 * progress + 0.6 * travel_distance + 0.5 * alignment

        if score > best_score:
            best_score = score
            best_waypoint = waypoint

    return best_waypoint


def plan_global_path_if_needed(robot, target):
    if target is None:
        reset_path(robot)
        return False

    target_id = robot.current_target
    goal_cell = nearest_free_cell(world_to_grid(target), OCCUPANCY_GRID)
    start_cell = nearest_free_cell(world_to_grid(robot_position(robot)), OCCUPANCY_GRID)

    if goal_cell is None or start_cell is None:
        reset_path(robot)
        return False

    should_replan = (
        robot.path_target_id != target_id
        or robot.path_goal_cell != goal_cell
        or not robot.global_path
        or robot.path_stall_steps >= PATH_STALL_THRESHOLD
    )

    if not should_replan:
        return True

    path_cells = astar_search(start_cell, goal_cell, OCCUPANCY_GRID)
    if not path_cells:
        reset_path(robot)
        return False

    simplified_cells = simplify_path_cells(path_cells, OCCUPANCY_GRID)
    world_path = [grid_to_world(cell) for cell in simplified_cells]

    if np.linalg.norm(world_path[-1] - target) > WAYPOINT_REACHED_RADIUS:
        world_path.append(target.copy())
    else:
        world_path[-1] = target.copy()

    robot.global_path = world_path
    robot.path_waypoint_idx = 0
    robot.path_target_id = target_id
    robot.path_goal_cell = goal_cell
    robot.path_stall_steps = 0
    robot.last_waypoint_distance = None
    return True


def update_waypoint_progress(robot, waypoint):
    distance = np.linalg.norm(waypoint - robot_position(robot))

    if robot.last_waypoint_distance is not None and distance < robot.last_waypoint_distance - TARGET_PROGRESS_EPS:
        robot.path_stall_steps = 0
    else:
        robot.path_stall_steps += 1

    robot.last_waypoint_distance = distance


def get_next_waypoint(robot, target):
    if not robot.global_path:
        return None

    pos = robot_position(robot)

    while robot.path_waypoint_idx < len(robot.global_path):
        waypoint = robot.global_path[robot.path_waypoint_idx]
        if np.linalg.norm(waypoint - pos) < WAYPOINT_REACHED_RADIUS:
            robot.path_waypoint_idx += 1
            robot.last_waypoint_distance = None
        else:
            break

    if robot.path_waypoint_idx >= len(robot.global_path):
        return target

    best_idx = robot.path_waypoint_idx
    current_cell = world_to_grid(pos)

    for idx in range(robot.path_waypoint_idx, len(robot.global_path)):
        waypoint_cell = world_to_grid(robot.global_path[idx])
        if line_of_sight_cells(current_cell, waypoint_cell, OCCUPANCY_GRID):
            best_idx = idx
        else:
            break

    robot.path_waypoint_idx = best_idx
    waypoint = robot.global_path[best_idx]
    update_waypoint_progress(robot, waypoint)
    return waypoint


def choose_navigation_goal(robot, target):
    if target is None:
        robot.last_target_distance = None
        robot.last_target_id = None
        robot.target_stall_steps = 0
        reset_detour(robot)
        reset_path(robot)
        return None

    pos = robot_position(robot)
    target_distance = np.linalg.norm(target - pos)
    target_id = robot.current_target

    if robot.last_target_id != target_id:
        robot.target_stall_steps = 0
        robot.last_target_distance = target_distance
        robot.last_target_id = target_id
        reset_detour(robot)
    else:
        if (
            robot.last_target_distance is not None
            and target_distance < robot.last_target_distance - TARGET_PROGRESS_EPS
        ):
            robot.target_stall_steps = 0
        else:
            robot.target_stall_steps += 1
        robot.last_target_distance = target_distance

    if plan_global_path_if_needed(robot, target):
        waypoint = get_next_waypoint(robot, target)
        if waypoint is not None:
            if robot.path_stall_steps >= PATH_STALL_THRESHOLD:
                reset_path(robot)
                if plan_global_path_if_needed(robot, target):
                    waypoint = get_next_waypoint(robot, target)
            if waypoint is not None:
                return waypoint

    if robot.lidar is None:
        return target

    direct_angle = np.arctan2(target[1] - pos[1], target[0] - pos[0])
    direct_clearance = clearance_for_direction(robot, direct_angle)
    direct_blocked = direct_clearance < min(target_distance, SAFE_CLEARANCE)

    if robot.detour_waypoint is not None and robot.detour_target_id == target_id:
        robot.detour_hold_steps = max(robot.detour_hold_steps - 1, 0)
        waypoint_distance = np.linalg.norm(robot.detour_waypoint - pos)

        if waypoint_distance < TARGET_CAPTURE_RADIUS or (
            not direct_blocked and robot.detour_hold_steps == 0
        ):
            reset_detour(robot)
        else:
            return robot.detour_waypoint

    if direct_blocked or robot.target_stall_steps >= TARGET_STALL_THRESHOLD:
        detour_waypoint = plan_detour_waypoint(robot, target)
        if detour_waypoint is not None:
            robot.detour_waypoint = detour_waypoint
            robot.detour_target_id = target_id
            robot.detour_hold_steps = DETOUR_HOLD_STEPS
            robot.target_stall_steps = 0
            return detour_waypoint

    reset_detour(robot)
    return target


def select_escape_direction(robot):
    if robot.lidar is None:
        theta = robot.state[2, 0]
        return np.array([-np.cos(theta), -np.sin(theta)])

    best_idx = int(np.argmax(robot.lidar.range_data))
    best_angle = robot.state[2, 0] - np.pi / 2 + robot.lidar.angle_list[best_idx]
    direction = np.array([np.cos(best_angle), np.sin(best_angle)])

    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        theta = robot.state[2, 0]
        return np.array([-np.cos(theta), -np.sin(theta)])

    return direction / norm


def compute_robot_repulsion(robot, robot_list):
    pos = robot_position(robot)
    repulsive = np.zeros(2)

    for other in robot_list:
        if other is robot:
            continue

        offset = pos - robot_position(other)
        distance = np.linalg.norm(offset)

        if distance < 1e-6 or distance >= REPULSION_RANGE:
            continue

        repulsive += offset / (max(distance, 0.1) ** 2)

    return repulsive


def compute_velocity(robot, target):
    if robot.escape_steps > 0:
        return ESCAPE_SPEED * robot.escape_direction

    navigation_goal = choose_navigation_goal(robot, target)
    if navigation_goal is None:
        return np.zeros(2)

    pos = robot_position(robot)
    delta = navigation_goal - pos
    distance = np.linalg.norm(delta)

    if distance < 1e-9:
        return np.zeros(2)

    attractive = delta / distance
    robot_repulsive = compute_robot_repulsion(robot, env.robot_list)
    desired_direction = attractive + REPULSION_GAIN * robot_repulsive

    desired_norm = np.linalg.norm(desired_direction)
    if desired_norm < 1e-9:
        desired_direction = attractive
    else:
        desired_direction = desired_direction / desired_norm

    safe_direction, forward_clearance = select_safe_direction(robot, desired_direction)

    switch_distance = CRUISE_SPEED / max(APPROACH_GAIN, 1e-6)
    target_speed = CRUISE_SPEED if distance > switch_distance else APPROACH_GAIN * distance

    if np.isfinite(forward_clearance):
        clearance_speed_limit = max(
            0.0, SPEED_CLEARANCE_GAIN * (forward_clearance - SPEED_CLEARANCE_OFFSET)
        )
        speed = min(target_speed, clearance_speed_limit)
    else:
        speed = target_speed

    if speed < 1e-6:
        return np.zeros(2)

    return speed * safe_direction


def update_visited_targets(robot_list, target_points):
    for robot in robot_list:
        pos = robot_position(robot)
        for target_idx, target in enumerate(target_points):
            if np.linalg.norm(pos - target) < TARGET_CAPTURE_RADIUS:
                robot.visited_points.add(target_idx)
                invalidate_current_target_if_needed(robot)


def global_visited_count(robot_list):
    global_visited = set()
    for robot in robot_list:
        global_visited |= robot.visited_points
    return len(global_visited)


def update_escape_state(robot_list):
    for robot in robot_list:
        current_pos = robot_position(robot)
        moved_distance = np.linalg.norm(current_pos - robot.last_position)
        has_target = robot.current_target is not None

        if robot.escape_steps > 0:
            robot.escape_steps -= 1
            if robot.escape_steps == 0:
                robot.escape_direction = np.zeros(2)
                robot.collision_flag = False
        elif has_target and moved_distance < STUCK_DISTANCE:
            robot.stuck_steps += 1
        else:
            robot.stuck_steps = 0

        if robot.collision_flag or robot.stuck_steps >= STUCK_STEP_THRESHOLD:
            robot.escape_direction = select_escape_direction(robot)
            robot.escape_steps = ESCAPE_STEPS
            robot.stuck_steps = 0
            robot.collision_flag = False

        robot.last_position = current_pos.copy()


OCCUPANCY_GRID = build_occupancy_grid()

target_points = generate_target_points(TARGET_COUNT, seed=TARGET_SEED)
target_colors = ["orange"] * len(target_points)
target_plot = None

setup_robot_memory(env.robot_list)
initialize_target_assignments(env.robot_list, target_points, centralized_init=CENTRALIZED_INIT)

for step in range(MAX_STEPS):
    update_visited_targets(env.robot_list, target_points)

    contact_components = build_contact_components(env.robot_list, env.components)
    contact_logs = rebalance_components(contact_components, target_points, step)

    for component, shared_count, assignment_sizes in contact_logs:
        print(
            f"{step * env.step_time:.1f}s contact {list(component)} "
            f"shared {shared_count}/{len(target_points)} targets "
            f"assignments={assignment_sizes}"
        )

    vel_list = []
    for robot in env.robot_list:
        robot.cal_lidar_range(env.components)
        target = choose_target(robot, target_points)
        vel_list.append(compute_velocity(robot, target))

    env.robot_step(vel_list, vel_type="omni", stop=False)
    for robot in env.robot_list:
        robot.collision_flag = False
    env.collision_check()

    update_visited_targets(env.robot_list, target_points)
    update_escape_state(env.robot_list)

    if step % RENDER_INTERVAL == 0:
        env.render(0.001, show_goal=False, show_text=False)

        if target_plot is None:
            target_plot = env.world_plot.ax.scatter(
                target_points[:, 0],
                target_points[:, 1],
                s=25,
                c=target_colors,
                edgecolors="black",
            )

        for target_idx in range(len(target_points)):
            visited = any(target_idx in robot.visited_points for robot in env.robot_list)
            target_colors[target_idx] = "gray" if visited else "orange"

        target_plot.set_color(target_colors)

    if global_visited_count(env.robot_list) == len(target_points):
        print(
            f"{step * env.step_time:.1f} seconds: Mission Success! "
            f"All targets were covered."
        )
        break
