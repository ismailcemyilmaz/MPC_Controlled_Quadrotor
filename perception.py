"""
perception.py
=============
Quadrotor MPC — Perception Layer
---------------------------------
Three-level perception pipeline:

  Level 1 — Gazebo Ground Truth   : model positions via gz transport / CLI
  Level 2 — 2D Lidar              : Gazebo lidar plugin -> clustering -> obstacles
  Level 3 — Static Map            : Known static obstacles (simplest, production)

Usage (import in quadrotor_mpc_client.py):
    from perception import PerceptionManager
    _perception = PerceptionManager(level=1)
    _perception.start()
    ...
    obstacles = _perception.get_obstacles()   # [(p_obs, R_obs), ...]
"""

import threading
import time
import math
import numpy as np
import subprocess
import json
import re
import socket
import struct
from collections import defaultdict
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
Obstacle = Tuple[np.ndarray, float]   # (p_obs [3,], R_obs)


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 1 — GAZEBO GROUND TRUTH
#  Reads model positions from Gazebo Ionic via gz-transport or gz CLI.
#  No real sensor needed; ideal for testing and development.
# ══════════════════════════════════════════════════════════════════════════════

class GazeboGroundTruth:
    """
    Tracks model names in the Gazebo world, returns their positions.

    Tries gz-transport Python binding first (fast, <1ms).
    Falls back to gz CLI (subprocess, ~10ms).
    If both fail, /world/default/pose/info topic is
    listened via raw UDP (Gazebo Ionic default port 11345).
    """

    def __init__(self, obstacle_models: List[Tuple[str, float]],
                 update_rate_hz: float = 10.0):
        """
        obstacle_models : [(model_name, radius), ...]
            Example: [('cylinder_1', 0.3), ('box_obstacle', 0.4)]
        update_rate_hz  : polling frequency
        """
        self.obstacle_models = obstacle_models
        self.update_period   = 1.0 / update_rate_hz
        self._obstacles: List[Obstacle] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread  = None
        self._method  = None   # 'transport', 'cli', 'unavailable'

        self._detect_method()

    def _detect_method(self):
        """Determine which gz interface is available."""
        # 1. gz-transport Python binding
        for ver in [14, 13, 12, 11, 10]:
            try:
                mod = __import__(f'gz.transport{ver}')
                self._gz_transport = mod
                self._method = 'transport'
                print(f"[perception] gz.transport{ver} Python binding found")
                return
            except ImportError:
                pass

        # 2. gz CLI
        try:
            r = subprocess.run(['gz', 'model', '--list'],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                self._method = 'cli'
                print("[perception] gz CLI found — model polling active")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._method = 'unavailable'
        print("[perception] WARNING: gz transport and CLI not found")
        print("[perception]   -> Use StaticObstacles or ManualInput instead")

    def _get_pose_cli(self, model_name: str) -> Optional[np.ndarray]:
        """Read model position via gz model CLI."""
        try:
            r = subprocess.run(
                ['gz', 'model', '-m', model_name, '-p'],
                capture_output=True, text=True, timeout=0.5
            )
            if r.returncode != 0:
                return None
            out = r.stdout
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', out)
            if len(nums) >= 3:
                return np.array([float(nums[0]),
                                 float(nums[1]),
                                 float(nums[2])])
        except subprocess.TimeoutExpired:
            pass
        return None

    def _get_pose_transport(self, model_name: str) -> Optional[np.ndarray]:
        """Read pose via gz-transport Python binding."""
        try:
            node = self._gz_transport.Node()
            msgs = []
            def cb(msg): msgs.append(msg)
            topic = '/world/default/pose/info'
            node.subscribe(topic, cb)
            time.sleep(0.05)
            for msg in msgs:
                for pose in msg.pose:
                    if pose.name == model_name:
                        p = pose.position
                        return np.array([p.x, p.y, p.z])
        except Exception:
            pass
        return None

    def _poll(self):
        """Background thread: periodically update obstacle list."""
        while self._running:
            new_obs = []
            for model_name, radius in self.obstacle_models:
                pos = None
                if self._method == 'transport':
                    pos = self._get_pose_transport(model_name)
                elif self._method == 'cli':
                    pos = self._get_pose_cli(model_name)

                if pos is not None:
                    new_obs.append((pos, radius))
                else:
                    print(f"[perception] WARNING: '{model_name}' not found")

            with self._lock:
                self._obstacles = new_obs

            time.sleep(self.update_period)

    def start(self):
        if self._method == 'unavailable':
            print("[perception] GazeboGroundTruth could not start")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        print(f"[perception] GazeboGroundTruth started "
              f"({len(self.obstacle_models)} models, {1/self.update_period:.0f} Hz)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_obstacles(self) -> List[Obstacle]:
        with self._lock:
            return list(self._obstacles)


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 2 — 2D LIDAR (Gazebo plugin)
#  Reads LaserScan from Gazebo gpu_lidar plugin.
#  Polar -> Cartesian -> DBSCAN clustering -> obstacle spheres
# ══════════════════════════════════════════════════════════════════════════════

class Lidar2DPerception:
    """
    Reads LaserScan from Gazebo 2D lidar plugin, clusters into obstacles.

    Drone SDF should include:
        <sensor name='lidar' type='gpu_lidar'>
          <topic>/lidar/scan</topic>
          <update_rate>10</update_rate>
          <ray>
            <scan>
              <horizontal>
                <samples>360</samples>
                <resolution>1</resolution>
                <min_angle>-3.14159</min_angle>
                <max_angle>3.14159</max_angle>
              </horizontal>
            </scan>
            <range>
              <min>0.08</min>
              <max>8.0</max>
            </range>
          </ray>
        </sensor>
    """

    def __init__(self,
                 topic: str      = '/lidar/scan',
                 drone_radius: float = 0.30,
                 max_range: float    = 6.0,
                 cluster_eps: float  = 0.25,   # DBSCAN epsilon [m]
                 cluster_min: int    = 3,       # min points/cluster
                 min_obs_r: float    = 0.15,    # min obstacle radius [m]
                 max_obs_r: float    = 1.50,    # max obstacle radius [m]
                 drone_height: float = 1.5):    # fixed height (2D)
        self.topic        = topic
        self.drone_radius = drone_radius
        self.max_range    = max_range
        self.cluster_eps  = cluster_eps
        self.cluster_min  = cluster_min
        self.min_obs_r    = min_obs_r
        self.max_obs_r    = max_obs_r
        self.drone_height = drone_height

        self._obstacles: List[Obstacle] = []
        self._lock    = threading.Lock()
        self._running = False
        self._subscriber = None

        self._drone_pos = np.zeros(3)

    def update_drone_pos(self, pos: np.ndarray):
        """Update drone position each step (for LaserScan transform)."""
        self._drone_pos = pos.copy()

    def _polar_to_cartesian(self, ranges, angle_min, angle_increment,
                            max_range_override=None):
        """LaserScan polar -> world frame cartesian points."""
        points = []
        dx, dy = self._drone_pos[0], self._drone_pos[1]
        mr = max_range_override if max_range_override else self.max_range
        for i, r in enumerate(ranges):
            if r < 0.1 or r > mr:
                continue
            angle = angle_min + i * angle_increment
            x = dx + r * math.cos(angle)
            y = dy + r * math.sin(angle)
            points.append([x, y])
        return np.array(points) if points else np.empty((0, 2))

    def _dbscan_cluster(self, points: np.ndarray):
        """Simple DBSCAN implementation (no sklearn dependency)."""
        if len(points) == 0:
            return []

        n      = len(points)
        labels = [-1] * n   # -1 = noise
        cluster_id = 0

        def neighbors(idx):
            dists = np.linalg.norm(points - points[idx], axis=1)
            return np.where(dists < self.cluster_eps)[0].tolist()

        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nb = neighbors(i)
            if len(nb) < self.cluster_min:
                labels[i] = -1   # noise
                continue
            labels[i] = cluster_id
            queue = list(nb)
            while queue:
                j = queue.pop(0)
                if not visited[j]:
                    visited[j] = True
                    nb2 = neighbors(j)
                    if len(nb2) >= self.cluster_min:
                        queue.extend(nb2)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        clusters = defaultdict(list)
        for i, lbl in enumerate(labels):
            if lbl >= 0:
                clusters[lbl].append(points[i])
        return [np.array(pts) for pts in clusters.values()]

    def _clusters_to_obstacles(self, clusters, drone_z: float):
        """Each cluster -> center + radius -> Obstacle tuple."""
        obs = []
        for pts in clusters:
            center = pts.mean(axis=0)
            radius = np.linalg.norm(pts - center, axis=1).max()
            radius = np.clip(radius + self.drone_radius,
                             self.min_obs_r, self.max_obs_r)
            p_obs = np.array([center[0], center[1], drone_z])
            obs.append((p_obs, float(radius)))
        return obs

    def process_scan(self, msg):
        """
        Process a LaserScan message.
        msg structure must match Gazebo LaserScan protobuf.
        """
        try:
            if self._drone_pos[2] < 1.5:
                return

            ranges        = list(msg.ranges)
            angle_min     = msg.angle_min
            angle_increment = msg.angle_step

            max_r = min(self.max_range, self._drone_pos[2] * 3.0)

            pts = self._polar_to_cartesian(ranges, angle_min, angle_increment,
                                           max_range_override=max_r)
            if len(pts) < self.cluster_min:
                return

            clusters = self._dbscan_cluster(pts)
            new_obs  = self._clusters_to_obstacles(
                clusters, self._drone_pos[2]
            )

            prev_count = len(self._obstacles)
            with self._lock:
                self._obstacles = new_obs

            if len(new_obs) != prev_count:
                print(f"[lidar] {len(new_obs)} obstacle(s) detected: "
                      + ", ".join(f"r={o[1]:.2f}m" for o in new_obs))
        except Exception as e:
            print(f"[lidar] scan processing error: {e}")

    def start(self):
        """Start gz-transport subscriber."""
        gz_t = None
        for ver in [14, 13, 12, 11, 10]:
            try:
                gz_t = __import__(f'gz.transport{ver}')
                gz_t = getattr(gz_t, f'transport{ver}')
                print(f"[perception] gz.transport{ver} found for Lidar2D")
                break
            except (ImportError, AttributeError):
                pass

        if gz_t is None:
            print("[perception] gz.transport not found — Lidar2D not operational")
            print("[perception]   -> Use GazeboGroundTruth or StaticObstacles instead")
            return

        from gz.msgs11.laserscan_pb2 import LaserScan
        self._LaserScan = LaserScan

        def _raw_cb(raw_msg):
            try:
                scan = self._LaserScan()
                scan.ParseFromString(raw_msg)
                self.process_scan(scan)
            except Exception as e:
                print(f"[lidar] parse error: {e}")

        try:
            node = gz_t.Node()
            ok = node.subscribe(LaserScan, self.topic, self.process_scan)
            if ok:
                self._subscriber = node
                self._running = True
                print(f"[perception] Lidar2D started — topic: {self.topic}")
            else:
                print(f"[perception] Lidar2D subscribe returned False — topic may not exist")
        except Exception as e:
            print(f"[perception] Lidar2D subscribe failed: {e}")
            print("[perception]   -> Check if Gazebo is running and lidar topic exists")

    def stop(self):
        self._running = False

    def get_obstacles(self) -> List[Obstacle]:
        with self._lock:
            return list(self._obstacles)


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 3 — STATIC OBSTACLES (manually defined)
#  Simplest, most reliable. For known environments.
# ══════════════════════════════════════════════════════════════════════════════

class StaticObstacles:
    """
    Fixed obstacles — positions entered manually from world file.
    No perception sensor needed.

    Example:
        static = StaticObstacles([
            {'pos': [2.0, 0.0, 1.5], 'radius': 0.4, 'name': 'cylinder_1'},
            {'pos': [4.0, 1.0, 1.0], 'radius': 0.3, 'name': 'box_1'},
        ])
    """

    def __init__(self, obstacles: list = None):
        self._obstacles: List[Obstacle] = []
        if obstacles:
            for obs in obstacles:
                p = np.array(obs['pos'], dtype=float)
                r = float(obs['radius'])
                self._obstacles.append((p, r))
        print(f"[perception] StaticObstacles: {len(self._obstacles)} obstacle(s) loaded")

    def start(self): pass
    def stop(self):  pass

    def add(self, x, y, z, radius):
        self._obstacles.append((np.array([x, y, z]), float(radius)))
        print(f"[perception] Obstacle added: ({x:.2f},{y:.2f},{z:.2f}) r={radius:.2f}m")

    def remove(self, idx):
        if 0 <= idx < len(self._obstacles):
            self._obstacles.pop(idx)

    def clear(self):
        self._obstacles.clear()

    def get_obstacles(self) -> List[Obstacle]:
        return list(self._obstacles)


# ══════════════════════════════════════════════════════════════════════════════
#  PERCEPTION MANAGER — unified interface
# ══════════════════════════════════════════════════════════════════════════════

class PerceptionManager:
    """
    Single-point interface for _run_loop in quadrotor_mpc_client.py.

    level=1 -> GazeboGroundTruth (gz CLI/transport)
    level=2 -> Lidar2DPerception
    level=3 -> StaticObstacles

    Example usage:
        perception = PerceptionManager(
            level=1,
            obstacle_models=[('obstacle_cylinder', 0.35),
                              ('obstacle_box',      0.40)],
        )
        perception.start()
        ...
        obstacles = perception.get_obstacles()   # pass to _run_loop
    """

    def __init__(self, level: int = 3, **kwargs):
        self.level = level

        if level == 1:
            models = kwargs.get('obstacle_models', [])
            rate   = kwargs.get('update_rate_hz', 10.0)
            self._backend = GazeboGroundTruth(models, rate)

        elif level == 2:
            self._backend = Lidar2DPerception(
                topic      = kwargs.get('topic', '/lidar/scan'),
                drone_radius = kwargs.get('drone_radius', 0.30),
                max_range  = kwargs.get('max_range', 6.0),
            )

        else:   # level 3 — default
            obs = kwargs.get('obstacles', [])
            self._backend = StaticObstacles(obs)

        print(f"[PerceptionManager] Level {level} active")

    def start(self):
        self._backend.start()

    def stop(self):
        self._backend.stop()

    def get_obstacles(self) -> List[Obstacle]:
        return self._backend.get_obstacles()

    def update_drone_pos(self, pos: np.ndarray):
        """Update drone position for Lidar2D transform."""
        if hasattr(self._backend, 'update_drone_pos'):
            self._backend.update_drone_pos(pos)

    def add_static(self, x, y, z, r):
        if hasattr(self._backend, 'add'):
            self._backend.add(x, y, z, r)

    def remove_static(self, idx):
        if hasattr(self._backend, 'remove'):
            self._backend.remove(idx)

    def clear_obstacles(self):
        if hasattr(self._backend, 'clear'):
            self._backend.clear()
            print("[perception] All obstacles cleared")
