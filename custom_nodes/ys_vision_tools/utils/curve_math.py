"""
Mathematical curve generation utilities for YS-vision-tools
Implements various parametric curves with proper mathematical equations
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.interpolate import CubicSpline, BSpline, splrep, splev
from scipy.spatial import distance_matrix, Voronoi, Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree


class CurveGenerator:
    """Generate various mathematical curves between two points"""

    def __init__(self, samples_per_curve: int = 50):
        """
        Initialize curve generator

        Args:
            samples_per_curve: Number of points to generate along each curve
        """
        self.samples = samples_per_curve

    def generate_straight(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Generate straight line

        Args:
            p1: Start point [x, y]
            p2: End point [x, y]

        Returns:
            Array of points along the line (N, 2)
        """
        t = np.linspace(0, 1, self.samples)
        return np.outer(1-t, p1) + np.outer(t, p2)

    def generate_quadratic_bezier(self, p1: np.ndarray, p2: np.ndarray,
                                  overshoot: float = 0.0) -> np.ndarray:
        """
        Generate quadratic Bezier curve: B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2

        Args:
            p1: Start point
            p2: End point
            overshoot: Control point overshoot factor (-1 to 1)

        Returns:
            Array of curve points (N, 2)
        """
        t = np.linspace(0, 1, self.samples)

        # Compute control point perpendicular to line
        mid = (p1 + p2) / 2
        normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        norm_length = np.linalg.norm(normal)

        if norm_length > 0:
            normal = normal / norm_length

        control = mid + normal * np.linalg.norm(p2 - p1) * (0.3 + overshoot)

        # Quadratic Bezier formula
        points = (np.outer((1-t)**2, p1) +
                 np.outer(2*(1-t)*t, control) +
                 np.outer(t**2, p2))

        return points

    def generate_cubic_bezier(self, p1: np.ndarray, p2: np.ndarray,
                              overshoot: float = 0.0,
                              control_offset: float = 0.3) -> np.ndarray:
        """
        Generate cubic Bezier curve: B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3

        Args:
            p1: Start point
            p2: End point
            overshoot: Control point overshoot factor
            control_offset: Position of control points along the line (0-1)

        Returns:
            Array of curve points (N, 2)
        """
        t = np.linspace(0, 1, self.samples)

        # Compute two control points
        v = p2 - p1
        normal = np.array([-v[1], v[0]])
        norm_length = np.linalg.norm(normal)

        if norm_length > 0:
            normal = normal / norm_length

        c1 = p1 + v * control_offset + normal * np.linalg.norm(v) * (0.2 + overshoot)
        c2 = p2 - v * control_offset + normal * np.linalg.norm(v) * (0.2 - overshoot)

        # Cubic Bezier formula
        points = (np.outer((1-t)**3, p1) +
                 np.outer(3*(1-t)**2*t, c1) +
                 np.outer(3*(1-t)*t**2, c2) +
                 np.outer(t**3, p2))

        return points

    def generate_catmull_rom(self, p1: np.ndarray, p2: np.ndarray,
                            tension: float = 0.5) -> np.ndarray:
        """
        Generate Catmull-Rom spline with tension control

        Args:
            p1: Start point
            p2: End point
            tension: Tension parameter (0 = no tension, 1 = tight)

        Returns:
            Array of curve points (N, 2)
        """
        t = np.linspace(0, 1, self.samples)

        # Create virtual points before and after
        p0 = p1 - (p2 - p1) * 0.5
        p3 = p2 + (p2 - p1) * 0.5

        # Catmull-Rom matrix with tension
        points = []
        for ti in t:
            t2 = ti * ti
            t3 = t2 * ti

            point = (
                (-tension*t3 + 2*tension*t2 - tension*ti) * p0 +
                ((2-tension)*t3 + (tension-3)*t2 + 1) * p1 +
                ((tension-2)*t3 + (3-2*tension)*t2 + tension*ti) * p2 +
                (tension*t3 - tension*t2) * p3
            )
            points.append(point)

        return np.array(points)

    def generate_logarithmic_spiral(self, p1: np.ndarray, p2: np.ndarray,
                                   turns: float = 0.5) -> np.ndarray:
        """
        Generate logarithmic spiral: r = a·exp(b·θ)

        Args:
            p1: Start point
            p2: End point
            turns: Number of spiral turns (0.5 = half turn)

        Returns:
            Array of curve points (N, 2)
        """
        # Convert to polar coordinates relative to p1
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        r_end = np.sqrt(dx**2 + dy**2)
        theta_end = np.arctan2(dy, dx)

        # Spiral parameters: r = a * exp(b * theta)
        a = 1.0
        b = np.log(r_end) / (theta_end + 2*np.pi*turns) if (theta_end + 2*np.pi*turns) != 0 else 0

        # Generate spiral
        theta = np.linspace(0, theta_end + 2*np.pi*turns, self.samples)
        r = a * np.exp(b * theta)

        # Convert back to Cartesian
        x = p1[0] + r * np.cos(theta)
        y = p1[1] + r * np.sin(theta)

        return np.column_stack([x, y])

    def generate_elastic_curve(self, p1: np.ndarray, p2: np.ndarray,
                               stiffness: float = 0.5) -> np.ndarray:
        """
        Generate elastic (damped oscillation) curve: y = A·sin(ωt)·exp(-γt)

        Args:
            p1: Start point
            p2: End point
            stiffness: Elastic stiffness (0 = loose, 1 = stiff)

        Returns:
            Array of curve points (N, 2)
        """
        t = np.linspace(0, 1, self.samples)

        # Base line
        base = np.outer(1-t, p1) + np.outer(t, p2)

        # Perpendicular direction
        v = p2 - p1
        normal = np.array([-v[1], v[0]])
        norm_length = np.linalg.norm(normal)

        if norm_length > 0:
            normal = normal / norm_length

        # Damped oscillation
        oscillation = np.sin(t * np.pi * 4) * (1 - t) * stiffness * 20

        # Apply oscillation perpendicular to base line
        offset = np.outer(oscillation, normal)
        points = base + offset

        return points

    def generate_fourier_series(self, p1: np.ndarray, p2: np.ndarray,
                                n_harmonics: int = 5) -> np.ndarray:
        """
        Generate curve using Fourier series approximation: f(t) = Σ(aₙ·sin(nωt) + bₙ·cos(nωt))

        Args:
            p1: Start point
            p2: End point
            n_harmonics: Number of Fourier harmonics

        Returns:
            Array of curve points (N, 2)
        """
        t = np.linspace(0, 1, self.samples)

        # Base interpolation
        base = np.outer(1-t, p1) + np.outer(t, p2)

        # Perpendicular direction
        v = p2 - p1
        normal = np.array([-v[1], v[0]])
        norm_length = np.linalg.norm(normal)

        if norm_length > 0:
            normal = normal / norm_length

        # Add Fourier harmonics
        fourier_sum = np.zeros(len(t))
        for n in range(1, n_harmonics + 1):
            amplitude = 1.0 / n
            freq = n * 2 * np.pi
            fourier_sum += amplitude * np.sin(freq * t)

        # Apply Fourier modulation
        offset = np.outer(fourier_sum * 10, normal)
        points = base + offset

        return points

    def generate_field_lines(self, p1: np.ndarray, p2: np.ndarray,
                            field_strength: float = 1.0) -> np.ndarray:
        """
        Generate magnetic/electric field line simulation

        Args:
            p1: Start point (source)
            p2: End point (target)
            field_strength: Field influence strength

        Returns:
            Array of curve points (N, 2)
        """
        points = [p1.copy()]
        current = p1.copy()

        for _ in range(self.samples - 1):
            # Vector towards target
            to_target = p2 - current
            distance = np.linalg.norm(to_target)

            if distance < 1e-6:
                points.append(current.copy())
                continue

            # Field direction (perpendicular component)
            direct_dir = to_target / distance
            field_dir = np.array([-to_target[1], to_target[0]])
            field_norm = np.linalg.norm(field_dir)

            if field_norm > 0:
                field_dir = field_dir / field_norm

            # Combine direct path with field influence
            step_size = distance / (self.samples - len(points))
            field_influence = field_dir * field_strength * np.sin(distance / 50) * 2

            current = current + direct_dir * step_size + field_influence
            points.append(current.copy())

        # Ensure we end at p2
        points[-1] = p2

        return np.array(points)

    def generate_gravitational_path(self, p1: np.ndarray, p2: np.ndarray,
                                   gravity_strength: float = 0.1) -> np.ndarray:
        """
        Generate gravitational path (parabolic trajectory)

        Args:
            p1: Start point
            p2: End point
            gravity_strength: Gravity influence (0-1)

        Returns:
            Array of curve points (N, 2)
        """
        t = np.linspace(0, 1, self.samples)

        # Linear interpolation
        base_x = p1[0] * (1-t) + p2[0] * t
        base_y = p1[1] * (1-t) + p2[1] * t

        # Add parabolic drop
        gravity_drop = -4 * t * (1 - t) * np.linalg.norm(p2 - p1) * gravity_strength

        x = base_x
        y = base_y + gravity_drop

        return np.column_stack([x, y])


class GraphBuilder:
    """Build various graph structures from point sets"""

    @staticmethod
    def build_knn_graph(points: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
        """
        Build k-nearest neighbor graph

        Args:
            points: Array of points (N, 2)
            k: Number of nearest neighbors

        Returns:
            List of edges as (i, j) tuples
        """
        from scipy.spatial import KDTree

        tree = KDTree(points)
        edges = set()

        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=min(k+1, len(points)))

            for j in indices[1:]:  # Skip self
                edge = tuple(sorted([i, j]))
                edges.add(edge)

        return list(edges)

    @staticmethod
    def build_radius_graph(points: np.ndarray, radius: float) -> List[Tuple[int, int]]:
        """
        Build radius-based graph

        Args:
            points: Array of points (N, 2)
            radius: Connection radius

        Returns:
            List of edges as (i, j) tuples
        """
        from scipy.spatial import KDTree

        tree = KDTree(points)
        edges = set()

        for i, point in enumerate(points):
            indices = tree.query_ball_point(point, radius)

            for j in indices:
                if i < j:
                    edges.add((i, j))

        return list(edges)

    @staticmethod
    def build_delaunay_graph(points: np.ndarray) -> List[Tuple[int, int]]:
        """
        Build Delaunay triangulation graph

        Args:
            points: Array of points (N, 2)

        Returns:
            List of edges as (i, j) tuples
        """
        if len(points) < 3:
            return []

        tri = Delaunay(points)
        edges = set()

        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)

        return list(edges)

    @staticmethod
    def build_mst_graph(points: np.ndarray) -> List[Tuple[int, int]]:
        """
        Build minimum spanning tree graph

        Args:
            points: Array of points (N, 2)

        Returns:
            List of edges as (i, j) tuples
        """
        if len(points) < 2:
            return []

        dist_matrix = distance_matrix(points, points)
        mst = minimum_spanning_tree(dist_matrix)
        edges = []

        rows, cols = mst.nonzero()
        for i, j in zip(rows, cols):
            if i < j:
                edges.append((i, j))

        return edges

    @staticmethod
    def build_voronoi_graph(points: np.ndarray) -> List[Tuple[int, int]]:
        """
        Build Voronoi diagram edges

        Args:
            points: Array of points (N, 2)

        Returns:
            List of edges as (i, j) tuples
        """
        if len(points) < 4:
            return []

        vor = Voronoi(points)
        edges = []

        for ridge in vor.ridge_points:
            if ridge[0] >= 0 and ridge[1] >= 0:
                edges.append(tuple(ridge))

        return edges
