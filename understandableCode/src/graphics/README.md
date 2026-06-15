# graphics

`graphics` is a local graphics, simulation, and game-helper package for UnderstandableCode.
It is meant to support both simple 2D work and more advanced scene-based 3D-style workflows
without depending on a big external engine.

## What this package is for

- drawing basic 2D shapes
- managing colors, vectors, and transforms
- building simple scene graphs
- projecting 3D points into 2D views
- running lightweight game state updates
- stepping basic physics and particle simulations

## Most used helpers

### Core data types

- `Color(red, green, blue, alpha=255)`
- `Vector2(x, y)`
- `Vector3(x, y, z)`
- `Transform2D`
- `Transform3D`
- `Sprite`
- `Mesh`
- `Camera`
- `SceneNode`
- `Canvas2D`

### Creation helpers

- `create_color(red, green, blue, alpha=255)`
- `create_vector2(x, y)`
- `create_vector3(x, y, z)`
- `create_transform_2d(x, y, rotation_degrees, scale_x, scale_y)`
- `create_transform_3d(x, y, z, rotation_x_degrees, rotation_y_degrees, rotation_z_degrees, scale_x, scale_y, scale_z)`
- `create_canvas_2d(width, height, background_color=None)`
- `create_sprite(name, width, height, color=None)`
- `create_mesh(name, vertices, faces)`
- `create_camera(position=None, rotation_degrees=None, field_of_view_degrees=60.0)`
- `create_scene_node(name, transform_2d=None, transform_3d=None, sprite=None, mesh=None)`

### 2D drawing

- `Canvas2D.clear(color=None)`
- `Canvas2D.set_pixel(x, y, color)`
- `Canvas2D.draw_line(start, end, color)`
- `Canvas2D.draw_rectangle(top_left, width, height, color, filled=False)`
- `Canvas2D.draw_circle(center, radius, color, filled=False)`
- `Canvas2D.draw_sprite(sprite, position)`

### Scene and game helpers

- `build_scene_graph(root, nodes)`
- `create_2d_game_object(name, x, y, width, height, color=None)`
- `update_2d_game_object_position(node, delta_x, delta_y)`
- `create_game_state(name, objects=None)`
- `advance_game_state(game_state, delta_seconds)`
- `summarize_scene(scene)`

### 3D and simulation helpers

- `project_3d_point_to_2d(point, camera, viewport_width, viewport_height)`
- `create_physics_body_2d(position, velocity, mass=1.0, radius=1.0)`
- `apply_force_to_body_2d(body, force)`
- `step_physics_body_2d(body, delta_seconds, gravity=(0.0, 0.0))`
- `detect_circle_collision(first_position, first_radius, second_position, second_radius)`
- `create_particle_emitter(position, emission_rate=10.0, particle_lifetime=1.0)`
- `update_particle_emitter(emitter, delta_seconds)`

## Examples

### Draw a simple 2D scene

```python
from graphics.graphics_library import create_canvas_2d, create_color, create_sprite, create_vector2

canvas = create_canvas_2d(64, 32)
red = create_color(255, 0, 0)
sprite = create_sprite("block", 8, 8, red)
canvas.draw_sprite(sprite, create_vector2(10, 10))
```

### Make a small game object

```python
from graphics.graphics_library import create_2d_game_object, update_2d_game_object_position

player = create_2d_game_object("player", 4, 4, 8, 8)
player = update_2d_game_object_position(player, 1, 0)
```

### Project 3D to 2D

```python
from graphics.graphics_library import create_camera, create_vector3, project_3d_point_to_2d

camera = create_camera()
screen_point = project_3d_point_to_2d(create_vector3(1.0, 1.0, 3.0), camera, 800, 600)
```

## Advanced capabilities

The package also includes deeper scene, lighting, particle, collision, and timing helpers
that are designed to support richer simulation workflows while staying readable.

## Design goal

The package is meant to be a practical local foundation for:

- 2D tools
- 3D projection and scene organization
- small game prototypes
- simulation experiments
- future rendering and engine-style expansion
