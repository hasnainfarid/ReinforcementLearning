# Urban Traffic Environment

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import deque, defaultdict

# --- Configuration Defaults ---
DEFAULT_GRID_SIZE = 4
DEFAULT_LANES = 2
DEFAULT_NUM_PHASES = 2
DEFAULT_MAX_QUEUE = 20
DEFAULT_MAX_WAIT = 300
DEFAULT_STEP_SEC = 1
DEFAULT_EPISODE_SEC = 3600
DEFAULT_SPAWN_RATE = 0.2  # vehicles per edge per second
DEFAULT_SPEED_LIMIT = 10.0  # m/s
DEFAULT_CAPACITY = 20
DEFAULT_WEATHER = "clear"
VEHICLE_TYPES = ["car", "bus", "emergency"]

# --- Helper Functions ---
def shortest_path(city, start, end):
    # Simple BFS for grid, can be replaced with A*
    from collections import deque
    queue = deque()
    queue.append((start, [start]))
    visited = set()
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in city.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return [start]

# --- Vehicle Class ---
class Vehicle:
    def __init__(self, vehicle_id, vehicle_type, origin, destination, route, spawn_time):
        self.id = vehicle_id
        self.type = vehicle_type
        self.origin = origin
        self.destination = destination
        self.route = route
        self.current_pos = origin
        self.next_pos = route[1] if len(route) > 1 else origin
        self.route_index = 0
        self.lane = 0
        self.speed = 0.0
        self.wait_time = 0
        self.total_time = 0
        self.completed = False
        self.spawn_time = spawn_time
        self.is_waiting = False
        self.is_emergency = (vehicle_type == "emergency")

# --- Traffic Light Class ---
class TrafficLight:
    def __init__(self, num_phases=DEFAULT_NUM_PHASES):
        self.num_phases = num_phases
        self.current_phase = 0
        self.phase_duration = [30] * num_phases
        self.time_in_phase = 0
        self.last_change = 0

    def step(self, action, dt):
        changed = False
        if action != self.current_phase:
            self.current_phase = action
            self.time_in_phase = 0
            changed = True
        else:
            self.time_in_phase += dt
        return changed

# --- Intersection Class ---
class Intersection:
    def __init__(self, pos, num_phases=DEFAULT_NUM_PHASES):
        self.pos = pos
        self.traffic_light = TrafficLight(num_phases)
        self.queues = {d: deque() for d in range(4)}  # 0:N, 1:E, 2:S, 3:W
        self.waiting_times = {d: deque() for d in range(4)}
        self.emergency_approach = [0, 0, 0, 0]

# --- Road Class ---
class Road:
    def __init__(self, start, end, lanes=DEFAULT_LANES, speed_limit=DEFAULT_SPEED_LIMIT, capacity=DEFAULT_CAPACITY):
        self.start = start
        self.end = end
        self.lanes = lanes
        self.speed_limit = speed_limit
        self.capacity = capacity
        self.vehicles = [[] for _ in range(lanes)]  # List of vehicles per lane

# --- City Class ---
class City:
    def __init__(self, grid_size=DEFAULT_GRID_SIZE, lanes=DEFAULT_LANES):
        self.grid_size = grid_size
        self.lanes = lanes
        self.intersections = {}
        self.roads = {}
        for i in range(grid_size):
            for j in range(grid_size):
                self.intersections[(i, j)] = Intersection((i, j))
        for i in range(grid_size):
            for j in range(grid_size):
                if i < grid_size - 1:
                    self.roads[((i, j), (i+1, j))] = Road((i, j), (i+1, j), lanes)
                    self.roads[((i+1, j), (i, j))] = Road((i+1, j), (i, j), lanes)
                if j < grid_size - 1:
                    self.roads[((i, j), (i, j+1))] = Road((i, j), (i, j+1), lanes)
                    self.roads[((i, j+1), (i, j))] = Road((i, j+1), (i, j), lanes)

    def get_neighbors(self, pos):
        i, j = pos
        neighbors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                neighbors.append((ni, nj))
        return neighbors

# --- Main Environment ---
class UrbanTrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config=None):
        config = config or {}
        self.grid_size = config.get("grid_size", DEFAULT_GRID_SIZE)
        self.lanes = config.get("lanes", DEFAULT_LANES)
        self.num_phases = config.get("num_phases", DEFAULT_NUM_PHASES)
        self.max_queue = config.get("max_queue", DEFAULT_MAX_QUEUE)
        self.max_wait = config.get("max_wait", DEFAULT_MAX_WAIT)
        self.episode_sec = config.get("episode_sec", DEFAULT_EPISODE_SEC)
        self.step_sec = config.get("step_sec", DEFAULT_STEP_SEC)
        self.spawn_rate = config.get("spawn_rate", DEFAULT_SPAWN_RATE)
        self.weather = config.get("weather", DEFAULT_WEATHER)
        self.seed_val = config.get("seed", None)
        self.random = np.random.RandomState(self.seed_val)
        self.city = City(self.grid_size, self.lanes)
        self.vehicles = {}
        self.vehicle_id_counter = 0
        self.time = 0
        self.done = False
        self.screen = None
        self.clock = None
        self.render_mode = config.get("render_mode", None)
        self.window_size = 700
        self.font = None
        self.metrics = defaultdict(list)
        self._setup_spaces()

    def _setup_spaces(self):
        N = self.grid_size
        num_phases = self.num_phases
        self.observation_space = spaces.Dict({
            "traffic_density": spaces.Box(0, 1, shape=(N, N, 4), dtype=np.float32),
            "queue_lengths": spaces.Box(0, self.max_queue, shape=(N, N, 4), dtype=np.int32),
            "waiting_times": spaces.Box(0, self.max_wait, shape=(N, N, 4), dtype=np.float32),
            "traffic_lights": spaces.MultiDiscrete([num_phases] * (N * N)),
            "emergency_vehicles": spaces.Box(0, 1, shape=(N, N, 4), dtype=np.int32),
            "current_time": spaces.Box(0, 24, shape=(1,), dtype=np.float32),
            "congestion_level": spaces.Box(0, 1, shape=(N, N), dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([num_phases] * (N * N))

    def reset(self, seed=None, options=None):
        self.city = City(self.grid_size, self.lanes)
        self.vehicles = {}
        self.vehicle_id_counter = 0
        self.time = 0
        self.done = False
        self.metrics = defaultdict(list)
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        # 1. Apply traffic light actions
        light_changes = 0
        for idx, pos in enumerate(self.city.intersections):
            changed = self.city.intersections[pos].traffic_light.step(action[idx], self.step_sec)
            if changed:
                light_changes += 1

        # 2. Spawn vehicles at edges
        for edge in self._edge_positions():
            if self.random.rand() < self.spawn_rate:
                dest = self._random_inner_position()
                vtype = self._random_vehicle_type()
                route = shortest_path(self.city, edge, dest)
                vehicle = Vehicle(self.vehicle_id_counter, vtype, edge, dest, route, self.time)
                self.vehicles[self.vehicle_id_counter] = vehicle
                self.vehicle_id_counter += 1

        # 3. Move vehicles
        completed, stopped, over_time, emergency_delay = 0, 0, 0, 0
        for vid, vehicle in list(self.vehicles.items()):
            if vehicle.completed:
                continue
            # Move along route
            if vehicle.current_pos == vehicle.destination:
                vehicle.completed = True
                completed += 1
                continue
            # Simulate waiting at intersection
            intersection = self.city.intersections[vehicle.current_pos]
            approach = self._get_approach(vehicle)
            if intersection.traffic_light.current_phase == approach:
                # Move to next position
                vehicle.route_index += 1
                if vehicle.route_index < len(vehicle.route):
                    vehicle.current_pos = vehicle.route[vehicle.route_index]
                else:
                    vehicle.completed = True
                    completed += 1
            else:
                vehicle.wait_time += self.step_sec
                stopped += 1
                if vehicle.is_emergency:
                    emergency_delay += self.step_sec
            vehicle.total_time += self.step_sec
            if vehicle.total_time > self.max_wait:
                over_time += 1
                vehicle.completed = True

        # 4. Remove completed vehicles
        self.vehicles = {vid: v for vid, v in self.vehicles.items() if not v.completed}

        # 5. Update time
        self.time += self.step_sec
        obs = self._get_obs()
        reward = self._compute_reward(completed, stopped, over_time, emergency_delay, light_changes)
        terminated = self.time >= self.episode_sec
        truncated = False
        info = {"completed": completed, "stopped": stopped}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        N = self.grid_size
        density = np.zeros((N, N, 4), dtype=np.float32)
        queue_lengths = np.zeros((N, N, 4), dtype=np.int32)
        waiting_times = np.zeros((N, N, 4), dtype=np.float32)
        traffic_lights = []
        emergency_vehicles = np.zeros((N, N, 4), dtype=np.int32)
        congestion_level = np.zeros((N, N), dtype=np.float32)
        for (i, j), intersection in self.city.intersections.items():
            for d in range(4):
                queue_lengths[i, j, d] = len(intersection.queues[d])
                waiting_times[i, j, d] = np.mean(intersection.waiting_times[d]) if intersection.waiting_times[d] else 0
                emergency_vehicles[i, j, d] = intersection.emergency_approach[d]
                density[i, j, d] = min(1.0, len(intersection.queues[d]) / self.max_queue)
            traffic_lights.append(intersection.traffic_light.current_phase)
            congestion_level[i, j] = np.sum(queue_lengths[i, j]) / (self.max_queue * 4)
        obs = {
            "traffic_density": density,
            "queue_lengths": queue_lengths,
            "waiting_times": waiting_times,
            "traffic_lights": np.array(traffic_lights, dtype=np.int32),
            "emergency_vehicles": emergency_vehicles,
            "current_time": np.array([self.time / 3600 * 24], dtype=np.float32),
            "congestion_level": congestion_level,
        }
        return obs

    def _compute_reward(self, completed, stopped, over_time, emergency_delay, light_changes):
        avg_wait = np.mean([v.wait_time for v in self.vehicles.values()]) if self.vehicles else 0
        total_congestion = np.sum([len(v.route) for v in self.vehicles.values()])
        balance_metric = 1.0  # Placeholder for balanced flow
        reward = (
            -0.01 * avg_wait
            -0.005 * stopped
            -0.1 * over_time
            +1.0 * completed
            -0.5 * emergency_delay
            -0.2 * light_changes
            -0.001 * total_congestion
            +0.1 * balance_metric
        )
        return reward

    def _edge_positions(self):
        N = self.grid_size
        positions = []
        for i in range(N):
            positions.append((i, 0))
            positions.append((i, N-1))
            positions.append((0, i))
            positions.append((N-1, i))
        return list(set(positions))

    def _random_inner_position(self):
        N = self.grid_size
        return (self.random.randint(1, N-1), self.random.randint(1, N-1))

    def _random_vehicle_type(self):
        return self.random.choice(VEHICLE_TYPES, p=[0.85, 0.1, 0.05])

    def _get_approach(self, vehicle):
        # Returns direction index (0:N, 1:E, 2:S, 3:W) for vehicle at intersection
        if vehicle.route_index == 0:
            return self.random.randint(0, 4)
        prev = vehicle.route[vehicle.route_index-1]
        curr = vehicle.current_pos
        di, dj = curr[0] - prev[0], curr[1] - prev[1]
        if di == -1: return 0
        if dj == 1: return 1
        if di == 1: return 2
        if dj == -1: return 3
        return 0

    # --- Pygame Visualization ---
    def render(self):
        N = self.grid_size
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Urban Traffic Simulator")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
        self.screen.fill((220, 220, 220))
        cell = self.window_size // N
        # Draw roads
        for i in range(N):
            for j in range(N):
                x, y = j * cell, i * cell
                pygame.draw.rect(self.screen, (180, 180, 180), (x+cell//4, y, cell//2, cell))
                pygame.draw.rect(self.screen, (180, 180, 180), (x, y+cell//4, cell, cell//2))
        # Draw intersections and traffic lights
        for (i, j), intersection in self.city.intersections.items():
            x, y = j * cell + cell//2, i * cell + cell//2
            color = [(0,255,0),(255,0,0)][intersection.traffic_light.current_phase % 2]
            pygame.draw.circle(self.screen, color, (x, y), cell//6)
            txt = self.font.render(str(intersection.traffic_light.current_phase), True, (0,0,0))
            self.screen.blit(txt, (x-cell//8, y-cell//8))
        # Draw vehicles
        for v in self.vehicles.values():
            i, j = v.current_pos
            x, y = j * cell + cell//2, i * cell + cell//2
            if v.type == "car":
                color = (0,0,255)
            elif v.type == "bus":
                color = (255,165,0)
            else:
                color = (255,0,0)
            pygame.draw.circle(self.screen, color, (x, y), cell//8)
        # Draw metrics
        avg_wait = np.mean([v.wait_time for v in self.vehicles.values()]) if self.vehicles else 0
        txt = self.font.render(f"Time: {self.time}s  Vehicles: {len(self.vehicles)}  Avg Wait: {avg_wait:.1f}s", True, (0,0,0))
        self.screen.blit(txt, (10, 10))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

