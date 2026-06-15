"""Readable graphics, game, and simulation helpers.

The goal of this module is not to be a full engine, but to provide a practical
local foundation for:
- 2D drawing and scene composition
- 3D-style transforms and projection
- basic game state and animation loops
- simple physics and simulation stepping
- hidden advanced helpers for lighting, particles, and collision handling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Number = float


def clamp_value(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def lerp_value(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


@dataclass
class Color:
    red: int
    green: int
    blue: int
    alpha: int = 255

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.red, self.green, self.blue, self.alpha

    def blend_with(self, other: "Color", amount: float) -> "Color":
        return Color(
            int(lerp_value(self.red, other.red, amount)),
            int(lerp_value(self.green, other.green, amount)),
            int(lerp_value(self.blue, other.blue, amount)),
            int(lerp_value(self.alpha, other.alpha, amount)),
        )


# Named color map for convenience
NAMED_COLORS = {
    "red": Color(255, 0, 0),
    "green": Color(0, 255, 0),
    "blue": Color(0, 0, 255),
    "white": Color(255, 255, 255),
    "black": Color(0, 0, 0),
    "yellow": Color(255, 255, 0),
    "cyan": Color(0, 255, 255),
    "magenta": Color(255, 0, 255),
    "orange": Color(255, 165, 0),
    "purple": Color(128, 0, 128),
    "pink": Color(255, 192, 203),
    "gray": Color(128, 128, 128),
    "grey": Color(128, 128, 128),
    "brown": Color(165, 42, 42),
    "lime": Color(0, 255, 0),
    "navy": Color(0, 0, 128),
    "teal": Color(0, 128, 128),
    "maroon": Color(128, 0, 0),
    "olive": Color(128, 128, 0),
    "darkgreen": Color(0, 100, 0),
}


def _resolve_color(color) -> Color:
    """Convert a string color name, tuple, or Color to a Color object."""
    if isinstance(color, Color):
        return color
    if isinstance(color, str):
        name = color.lower().replace(" ", "")
        if name in NAMED_COLORS:
            return NAMED_COLORS[name]
        # Try to parse as hex or named color
        return Color(255, 255, 255)  # default white
    if isinstance(color, (tuple, list)):
        if len(color) >= 3:
            return Color(int(color[0]), int(color[1]), int(color[2]), int(color[3]) if len(color) > 3 else 255)
    return Color(255, 255, 255)


@dataclass
class Vector2:
    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def scale(self, factor: float) -> "Vector2":
        return Vector2(self.x * factor, self.y * factor)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> "Vector2":
        magnitude = self.length() or 1.0
        return Vector2(self.x / magnitude, self.y / magnitude)


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vector3":
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> "Vector3":
        magnitude = self.length() or 1.0
        return Vector3(self.x / magnitude, self.y / magnitude, self.z / magnitude)


@dataclass
class Transform2D:
    position: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    rotation_degrees: float = 0.0
    scale: Vector2 = field(default_factory=lambda: Vector2(1.0, 1.0))


@dataclass
class Transform3D:
    position: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    rotation_degrees: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))


@dataclass
class Sprite:
    name: str
    width: int
    height: int
    color: Color = field(default_factory=lambda: Color(255, 255, 255))


@dataclass
class Mesh:
    name: str
    vertices: List[Vector3]
    faces: List[Tuple[int, int, int]]


@dataclass
class Camera:
    position: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, -5.0))
    rotation_degrees: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    field_of_view_degrees: float = 60.0
    near_clip: float = 0.1
    far_clip: float = 1000.0


@dataclass
class SceneNode:
    name: str
    transform_2d: Optional[Transform2D] = None
    transform_3d: Optional[Transform3D] = None
    sprite: Optional[Sprite] = None
    mesh: Optional[Mesh] = None
    children: List["SceneNode"] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_child(self, node: "SceneNode") -> None:
        self.children.append(node)


@dataclass
class Canvas2D:
    width: int
    height: int
    background_color: Color = field(default_factory=lambda: Color(0, 0, 0))
    pixels: List[List[Color]] = field(init=False)

    def __post_init__(self):
        self.clear()

    def clear(self, color=None) -> None:
        fill = _resolve_color(color) if color is not None else self.background_color
        self.pixels = [[fill for _ in range(self.width)] for _ in range(self.height)]

    def set_pixel(self, x: int, y: int, color) -> None:
        resolved = _resolve_color(color)
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = resolved

    def draw_line(self, start: Vector2, end: Vector2, color: Color) -> None:
        dx = abs(end.x - start.x)
        dy = -abs(end.y - start.y)
        step_x = 1 if start.x < end.x else -1
        step_y = 1 if start.y < end.y else -1
        error = dx + dy
        x = int(round(start.x))
        y = int(round(start.y))
        end_x = int(round(end.x))
        end_y = int(round(end.y))
        while True:
            self.set_pixel(x, y, color)
            if x == end_x and y == end_y:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x += step_x
            if doubled <= dx:
                error += dx
                y += step_y

    def draw_rectangle(self, top_left: Vector2, width: int, height: int, color: Color, filled: bool = False) -> None:
        if filled:
            for y in range(int(top_left.y), int(top_left.y) + height):
                for x in range(int(top_left.x), int(top_left.x) + width):
                    self.set_pixel(x, y, color)
            return
        self.draw_line(top_left, Vector2(top_left.x + width, top_left.y), color)
        self.draw_line(top_left, Vector2(top_left.x, top_left.y + height), color)
        self.draw_line(Vector2(top_left.x + width, top_left.y), Vector2(top_left.x + width, top_left.y + height), color)
        self.draw_line(Vector2(top_left.x, top_left.y + height), Vector2(top_left.x + width, top_left.y + height), color)

    def draw_circle(self, center: Vector2, radius: int, color: Color, filled: bool = False) -> None:
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                distance = math.sqrt(x * x + y * y)
                if filled and distance <= radius:
                    self.set_pixel(int(center.x + x), int(center.y + y), color)
                elif not filled and abs(distance - radius) <= 0.75:
                    self.set_pixel(int(center.x + x), int(center.y + y), color)

    def draw_sprite(self, sprite: Sprite, position: Vector2) -> None:
        self.draw_rectangle(position, sprite.width, sprite.height, sprite.color, filled=True)


def create_color(red: int, green: int, blue: int, alpha: int = 255) -> Color:
    return Color(red, green, blue, alpha)


def create_vector2(x: float, y: float) -> Vector2:
    return Vector2(x, y)


def create_vector3(x: float, y: float, z: float) -> Vector3:
    return Vector3(x, y, z)


def create_transform_2d(x: float = 0.0, y: float = 0.0, rotation_degrees: float = 0.0, scale_x: float = 1.0, scale_y: float = 1.0) -> Transform2D:
    return Transform2D(position=Vector2(x, y), rotation_degrees=rotation_degrees, scale=Vector2(scale_x, scale_y))


def create_transform_3d(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    rotation_x_degrees: float = 0.0,
    rotation_y_degrees: float = 0.0,
    rotation_z_degrees: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
) -> Transform3D:
    return Transform3D(
        position=Vector3(x, y, z),
        rotation_degrees=Vector3(rotation_x_degrees, rotation_y_degrees, rotation_z_degrees),
        scale=Vector3(scale_x, scale_y, scale_z),
    )


def create_canvas_2d(width: int, height: int, background_color: Optional[Color] = None) -> Canvas2D:
    return Canvas2D(width, height, background_color or Color(0, 0, 0))


def create_sprite(name: str, width: int, height: int, color: Optional[Color] = None) -> Sprite:
    return Sprite(name, width, height, color or Color(255, 255, 255))


def create_mesh(name: str, vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> Mesh:
    return Mesh(name, [Vector3(*vertex) for vertex in vertices], [tuple(face) for face in faces])


def create_camera(position: Optional[Sequence[float]] = None, rotation_degrees: Optional[Sequence[float]] = None, field_of_view_degrees: float = 60.0) -> Camera:
    position_values = position or (0.0, 0.0, -5.0)
    rotation_values = rotation_degrees or (0.0, 0.0, 0.0)
    return Camera(Vector3(*position_values), Vector3(*rotation_values), field_of_view_degrees)


def create_scene_node(name: str, transform_2d: Optional[Transform2D] = None, transform_3d: Optional[Transform3D] = None, sprite: Optional[Sprite] = None, mesh: Optional[Mesh] = None) -> SceneNode:
    return SceneNode(name, transform_2d, transform_3d, sprite, mesh)


def build_scene_graph(root: SceneNode, nodes: Iterable[SceneNode]) -> SceneNode:
    for node in nodes:
        root.add_child(node)
    return root


def project_3d_point_to_2d(point: Vector3, camera: Camera, viewport_width: int, viewport_height: int) -> Vector2:
    relative = point - camera.position
    depth = max(relative.z, camera.near_clip)
    aspect = viewport_width / max(viewport_height, 1)
    field_of_view_radians = math.radians(camera.field_of_view_degrees)
    focal_length = 1.0 / math.tan(field_of_view_radians / 2.0)
    projected_x = (relative.x * focal_length / depth) * aspect
    projected_y = (relative.y * focal_length / depth)
    screen_x = (projected_x + 1.0) * 0.5 * viewport_width
    screen_y = (1.0 - (projected_y + 1.0) * 0.5) * viewport_height
    return Vector2(screen_x, screen_y)


def create_2d_game_object(name: str, x: float, y: float, width: int, height: int, color: Optional[Color] = None) -> SceneNode:
    return create_scene_node(name, transform_2d=create_transform_2d(x, y), sprite=create_sprite(name, width, height, color))


def update_2d_game_object_position(node: SceneNode, delta_x: float, delta_y: float) -> SceneNode:
    if node.transform_2d is None:
        node.transform_2d = create_transform_2d()
    node.transform_2d.position = Vector2(node.transform_2d.position.x + delta_x, node.transform_2d.position.y + delta_y)
    return node


def create_game_state(name: str, objects: Optional[Iterable[SceneNode]] = None) -> Dict[str, object]:
    return {
        "name": name,
        "objects": list(objects or []),
        "frame": 0,
        "time_seconds": 0.0,
        "running": True,
    }


def advance_game_state(game_state: Dict[str, object], delta_seconds: float) -> Dict[str, object]:
    game_state = dict(game_state)
    game_state["frame"] = int(game_state.get("frame", 0)) + 1
    game_state["time_seconds"] = float(game_state.get("time_seconds", 0.0)) + delta_seconds
    return game_state


def create_physics_body_2d(position: Sequence[float], velocity: Sequence[float], mass: float = 1.0, radius: float = 1.0) -> Dict[str, object]:
    return {
        "position": Vector2(*position),
        "velocity": Vector2(*velocity),
        "mass": mass,
        "radius": radius,
        "forces": Vector2(0.0, 0.0),
    }


def apply_force_to_body_2d(body: Dict[str, object], force: Sequence[float]) -> Dict[str, object]:
    force_vector = Vector2(*force)
    body = dict(body)
    current_force = body.get("forces", Vector2(0.0, 0.0))
    body["forces"] = current_force + force_vector
    return body


def step_physics_body_2d(body: Dict[str, object], delta_seconds: float, gravity: Sequence[float] = (0.0, 0.0)) -> Dict[str, object]:
    body = dict(body)
    mass = float(body.get("mass", 1.0)) or 1.0
    velocity = body.get("velocity", Vector2(0.0, 0.0))
    position = body.get("position", Vector2(0.0, 0.0))
    forces = body.get("forces", Vector2(0.0, 0.0))
    gravity_vector = Vector2(*gravity)
    acceleration = Vector2((forces.x / mass) + gravity_vector.x, (forces.y / mass) + gravity_vector.y)
    velocity = Vector2(velocity.x + acceleration.x * delta_seconds, velocity.y + acceleration.y * delta_seconds)
    position = Vector2(position.x + velocity.x * delta_seconds, position.y + velocity.y * delta_seconds)
    body["velocity"] = velocity
    body["position"] = position
    body["forces"] = Vector2(0.0, 0.0)
    return body


def detect_circle_collision(first_position: Sequence[float], first_radius: float, second_position: Sequence[float], second_radius: float) -> bool:
    dx = first_position[0] - second_position[0]
    dy = first_position[1] - second_position[1]
    distance = math.sqrt(dx * dx + dy * dy)
    return distance <= (first_radius + second_radius)


def create_particle_emitter(position: Sequence[float], emission_rate: float = 10.0, particle_lifetime: float = 1.0) -> Dict[str, object]:
    return {
        "position": Vector2(*position),
        "emission_rate": emission_rate,
        "particle_lifetime": particle_lifetime,
        "particles": [],
        "accumulator": 0.0,
    }


def update_particle_emitter(emitter: Dict[str, object], delta_seconds: float) -> Dict[str, object]:
    emitter = dict(emitter)
    accumulator = float(emitter.get("accumulator", 0.0)) + delta_seconds * float(emitter.get("emission_rate", 1.0))
    particles = list(emitter.get("particles", []))
    spawn_count = int(accumulator)
    for _ in range(spawn_count):
        particles.append({
            "position": emitter["position"],
            "velocity": Vector2(0.0, -1.0),
            "age": 0.0,
            "lifetime": float(emitter.get("particle_lifetime", 1.0)),
        })
    accumulator -= spawn_count
    updated_particles = []
    for particle in particles:
        age = particle["age"] + delta_seconds
        if age > particle["lifetime"]:
            continue
        updated_particles.append({
            "position": Vector2(particle["position"].x + particle["velocity"].x * delta_seconds, particle["position"].y + particle["velocity"].y * delta_seconds),
            "velocity": particle["velocity"],
            "age": age,
            "lifetime": particle["lifetime"],
        })
    emitter["particles"] = updated_particles
    emitter["accumulator"] = accumulator
    return emitter


def compute_directional_light_intensity(surface_normal: Sequence[float], light_direction: Sequence[float], ambient_strength: float = 0.1) -> float:
    normal = Vector3(*surface_normal).normalize()
    light = Vector3(*light_direction).normalize()
    diffuse = max(0.0, normal.x * light.x + normal.y * light.y + normal.z * light.z)
    return clamp_value(ambient_strength + diffuse)


def create_frame_timeline(total_frames: int, frame_rate: float) -> List[Dict[str, float]]:
    timeline = []
    for frame in range(total_frames):
        timeline.append({"frame": float(frame), "time_seconds": frame / frame_rate})
    return timeline


def create_pathfinding_grid(width: int, height: int, blocked_cells: Optional[Iterable[Tuple[int, int]]] = None) -> List[List[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for x, y in blocked_cells or []:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 1
    return grid


def summarize_scene(scene: SceneNode) -> Dict[str, object]:
    return {
        "name": scene.name,
        "children": len(scene.children),
        "has_sprite": scene.sprite is not None,
        "has_mesh": scene.mesh is not None,
        "has_2d_transform": scene.transform_2d is not None,
        "has_3d_transform": scene.transform_3d is not None,
    }


# ============================================
# GAME-SUPPORTED WINDOW SYSTEM
# ============================================
# Features:
#   - graphics.get_keys()             → list of pressed key names
#   - graphics.get_key_pressed("w")   → yeah/nah (bool)
#   - graphics.delta_time()           → seconds since last frame
#   - graphics.frame_count()          → current frame number
#   - graphics.window_should_close()  → yeah/nah (True if X clicked)
#   - graphics.show(canvas)           → redraws the window
# ============================================

import time as _time

_WINDOW = None
_PIXEL_SIZE = 20  # each canvas "pixel" is drawn as a 20x20 rectangle

# Game-loop tracking
_LAST_FRAME_TIME = _time.perf_counter()
_FRAME_COUNT = 0
_DELTA_TIME = 0.0


def _get_tk():
    """Lazy-import tkinter only when needed."""
    import tkinter as _tk
    return _tk


def _build_key_name(event):
    """Convert a tkinter keysym into a clean lowercase key name."""
    keysym = getattr(event, "keysym", "")
    # Map special keys to readable names
    special = {
        "space": "space",
        "Return": "enter",
        "Escape": "escape",
        "Tab": "tab",
        "BackSpace": "backspace",
        "Delete": "delete",
        "Up": "up",
        "Down": "down",
        "Left": "left",
        "Right": "right",
        "Shift_L": "shift",
        "Shift_R": "shift",
        "Control_L": "control",
        "Control_R": "control",
        "Alt_L": "alt",
        "Alt_R": "alt",
    }
    name = special.get(keysym, keysym.lower())
    return name


def _on_key_press(event):
    """Called when any key is pressed inside the window."""
    if _WINDOW is None:
        return
    name = _build_key_name(event)
    _WINDOW["pressed_keys"].add(name)


def _on_key_release(event):
    """Called when any key is released inside the window."""
    if _WINDOW is None:
        return
    name = _build_key_name(event)
    _WINDOW["pressed_keys"].discard(name)


def _on_window_close():
    """Called when the user clicks the X button on the window."""
    global _WINDOW
    if _WINDOW is not None:
        _WINDOW["should_close"] = True
        # Don't destroy yet — let the user finish the current frame


def _create_window(width: int, height: int):
    """Create or reuse a game window. Returns None if the window was closed."""
    global _WINDOW, _LAST_FRAME_TIME

    # If window exists and hasn't been closed, just resize if needed and return it
    if _WINDOW is not None:
        try:
            _WINDOW["window"].wm_state()  # check if window still exists
            if not _WINDOW.get("should_close", False):
                # Resize if needed
                if (_WINDOW.get("grid_cols") != width or
                    _WINDOW.get("grid_rows") != height):
                    new_w = width * _PIXEL_SIZE
                    new_h = height * _PIXEL_SIZE
                    _WINDOW["window"].geometry(f"{new_w}x{new_h}")
                    _WINDOW["canvas"].config(width=new_w, height=new_h)
                    _WINDOW["grid_cols"] = width
                    _WINDOW["grid_rows"] = height
                    # Rebuild rectangle IDs for new size
                    cvs = _WINDOW["canvas"]
                    cvs.delete("all")
                    rect_ids = []
                    for row in range(height):
                        row_ids = []
                        y1 = row * _PIXEL_SIZE
                        y2 = y1 + _PIXEL_SIZE
                        for col in range(width):
                            x1 = col * _PIXEL_SIZE
                            x2 = x1 + _PIXEL_SIZE
                            rid = cvs.create_rectangle(x1, y1, x2, y2, outline="", width=0)
                            row_ids.append(rid)
                        rect_ids.append(row_ids)
                    _WINDOW["rect_ids"] = rect_ids
                return _WINDOW
        except Exception:
            _WINDOW = None  # window was destroyed

    # Reset game-loop tracking
    _LAST_FRAME_TIME = _time.perf_counter()
    _FRAME_COUNT = 0

    tk = _get_tk()
    win = tk.Tk()
    win.title("UnderstandableCode Graphics")
    win.resizable(False, False)

    cvs_w = width * _PIXEL_SIZE
    cvs_h = height * _PIXEL_SIZE
    canvas_widget = tk.Canvas(win, width=cvs_w, height=cvs_h, highlightthickness=0)
    canvas_widget.pack()
    canvas_widget.focus_set()  # so it captures keyboard events

    # Pre-create rectangle IDs for every cell
    rect_ids = []
    for row in range(height):
        row_ids = []
        y1 = row * _PIXEL_SIZE
        y2 = y1 + _PIXEL_SIZE
        for col in range(width):
            x1 = col * _PIXEL_SIZE
            x2 = x1 + _PIXEL_SIZE
            rid = canvas_widget.create_rectangle(x1, y1, x2, y2, outline="", width=0)
            row_ids.append(rid)
        rect_ids.append(row_ids)

    # Bind keyboard events
    canvas_widget.bind("<KeyPress>", _on_key_press)
    canvas_widget.bind("<KeyRelease>", _on_key_release)

    # Bind window close event
    win.protocol("WM_DELETE_WINDOW", _on_window_close)

    _WINDOW = {
        "window": win,
        "canvas": canvas_widget,
        "rect_ids": rect_ids,
        "grid_cols": width,
        "grid_rows": height,
        "pressed_keys": set(),
        "should_close": False,
    }
    win.update()
    return _WINDOW


def show(canvas: Canvas2D) -> None:
    """Render a Canvas2D to a real game window using tkinter.
    Each pixel becomes a colored 20x20 rectangle.
    Automatically updates delta_time and frame_count tracking.
    """
    global _WINDOW, _LAST_FRAME_TIME, _FRAME_COUNT, _DELTA_TIME

    now = _time.perf_counter()
    _DELTA_TIME = now - _LAST_FRAME_TIME
    _LAST_FRAME_TIME = now
    _FRAME_COUNT += 1

    win_data = _create_window(canvas.width, canvas.height)
    if win_data is None or win_data.get("should_close", False):
        return

    canvas_widget = win_data["canvas"]
    rect_ids = win_data.get("rect_ids")
    if rect_ids is None:
        return

    try:
        for y in range(canvas.height):
            for x in range(canvas.width):
                pix = canvas.pixels[y][x]
                r = max(0, min(255, pix.red))
                g = max(0, min(255, pix.green))
                b = max(0, min(255, pix.blue))
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                canvas_widget.itemconfig(rect_ids[y][x], fill=hex_color)
        win_data["window"].update()
    except Exception:
        pass  # window was closed or destroyed


# ============================================
# GAME LOOP FUNCTIONS
# ============================================

def get_keys() -> list:
    """Return a list of all keys currently held down.
    Key names are lowercase: 'w', 'a', 's', 'd', 'space', 'up', 'enter', etc.
    Returns an empty list if no keys are pressed or window is closed.
    """
    if _WINDOW is None:
        return []
    try:
        return list(_WINDOW.get("pressed_keys", set()))
    except Exception:
        return []


def get_key_pressed(key_name: str) -> bool:
    """Check if a specific key is currently held down.
    Example: get_key_pressed("w") returns yeah if W is pressed.
    Key names are lowercase. Works at all: 'w', 'space', 'up', 'enter', etc.
    """
    if _WINDOW is None:
        return False
    try:
        return key_name.lower() in _WINDOW.get("pressed_keys", set())
    except Exception:
        return False


def delta_time() -> float:
    """Return the time in seconds since the last frame.
    Use this to make movement speed independent of frame rate.
    Example:  x = x + speed * delta_time()
    """
    return _DELTA_TIME


def frame_count() -> int:
    """Return the number of frames drawn so far.
    Starts at 0 and increments by 1 each time show() is called.
    """
    return _FRAME_COUNT


def window_should_close() -> bool:
    """Return yeah when the user clicks the X button on the window.
    Returns nah if the window hasn't been created yet (first loop iteration).
    Use this to exit your game loop gracefully.
    Example:
        loop while window_should_close() is nah
            ...
        end
    """
    if _WINDOW is None:
        return False  # Window not created yet — first frame, keep going
    try:
        return _WINDOW.get("should_close", False)
    except Exception:
        return True


def close_window() -> None:
    """Destroy the game window and clean up. Call this after your game loop ends."""
    global _WINDOW
    if _WINDOW is not None:
        try:
            _WINDOW["window"].destroy()
        except Exception:
            pass
        _WINDOW = None
