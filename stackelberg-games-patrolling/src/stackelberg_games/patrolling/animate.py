from __future__ import annotations

import math
from pathlib import Path
import pickle

import manim
import networkx
import numpy
import pydantic
import shapely
from termcolor import colored

from .mts import MapTileService
from .pwl import PWLMap
from .schedule import Schedule
from .setting import Rational


defaults = {
    'quality': 'high_quality',    # fourk_quality, production_quality, high_quality, example_quality
    'map_tile_service': 'osm',
    'background_color': manim.GRAY_A,
    'colors': {
        'osm': {
            'unprotected_fill': manim.BLACK,
            'protected_fill': manim.WHITE,
            'unprotected_border': manim.DARK_GRAY,
            'protected_border': manim.LIGHT_GRAY,
        },
        'google_terrain': {
            'unprotected_fill': manim.BLACK,
            'protected_fill': manim.WHITE,
            'unprotected_border': manim.DARK_GRAY,
            'protected_border': manim.LIGHT_GRAY,
        },
        'stadia_smooth': {
            'unprotected_fill': manim.WHITE,
            'protected_fill': manim.BLUE_E,
            'unprotected_border': manim.LIGHT_GRAY,
            'protected_border': manim.DARK_GRAY,
        }
    },
    'use_map': True,
    'map_zoom_level': 16,
    'defense_plan': 'gdynia_defense_plans/gdynia_1000_1_2_3.pickle',
    'steps': 10,
    'speed': 1,
    'rate': manim.linear
}

for color_tag in ['unprotected_fill', 'protected_fill', 'unprotected_border', 'protected_border']:
    defaults[color_tag] = defaults['colors'][defaults['map_tile_service']][color_tag]

if defaults['quality'] == 'example_quality':
    defaults['map_zoom_level'] -= 2


class Animation(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    map_tag: str
    """Map tile provider tag, as declared in mts.MapTileService."""
    defense_plan_filename: str
    """A filename of a pickled plan.DefensePlan."""

    random_seed: int | None = None
    """Random seed used to generate a schedule."""
    steps: int = 60
    """Number of rollout steps."""
    speed: Rational = 3
    """Number of seconds per animation step."""
    output_filename: str = None
    """Name of output file. If None, default manim filename is used."""

    def render(self):
        """Renders the animation."""
        with open(self.defense_plan_filename, 'rb') as input_file:
            defense_plan = pickle.load(input_file)

        rollout = defense_plan.generate_schedule(self.steps, random_seed=self.random_seed)

        with manim.tempconfig({"quality": defaults['quality'], "preview": False,
                               "output_file": f"{self.output_filename}", "flush_cache": True}):
            scene = ScheduleAnimation(rollout,
                                      map_tile_service=self.map_tag,
                                      speed=self.speed)
            scene.render()


class AnimationGroup(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    map_tags: list[str]
    """A list of map tile provider tags."""
    defense_plan_filenames: list[str]
    """A list of defense plans to process."""
    steps: list[int]
    """Number of rollout steps."""
    speeds: list[Rational] = 3
    """A list of animation speeds."""
    random_seed: int = None
    """Random seed."""

    @pydantic.computed_field
    @property
    def animations(self) -> list[Animation]:
        animation_list = []
        for map_tag in self.map_tags:
            for defense_plan_filename in self.defense_plan_filenames:
                for speed, steps in zip(self.speeds, self.steps):
                    tag = Path(defense_plan_filename).stem

                    animation_list.append(
                        Animation(
                            map_tag=map_tag,
                            defense_plan_filename=defense_plan_filename,
                            random_seed=self.random_seed,
                            steps=steps,
                            speed=speed,
                            output_filename=f'{map_tag}_{tag}_speed_{speed}.mp4'
                        )
                    )
        return animation_list


class ScheduleAnimation(manim.MovingCameraScene):
    def __init__(self, rollout: Schedule, map_tile_service: str = None, speed: int = None):
        super().__init__()

        self.rollout = rollout
        self.zoom_level = defaults['map_zoom_level']

        if map_tile_service is None:
            map_tile_service = defaults['map_tile_service']
        if speed is None:
            speed = defaults['speed']

        self.map_tile_service = map_tile_service
        self.speed = speed

        try:
            self.mts = MapTileService(map_tile_service)
        except ValueError as e:
            print(f'{e}. {colored("Falling back to OSM tile provider.", "red")}.')
            self.mts = MapTileService('osm')

        print(f'TILE SIZE {self.mts.get_tile_size(self.zoom_level)}')

        self.positions = {}
        for factor_space in self.rollout.data.base.factors:
            factor_graph = factor_space.topology
            for node, position in networkx.get_node_attributes(factor_graph, 'pos').items():
                point = numpy.array([position[0], position[1], 0])
                if node in self.positions and not numpy.allclose(self.positions[node], point):
                    raise ValueError('Conflicting nodes in factor graphs.')
                self.positions[node] = point

        geometry = shapely.MultiPoint(list(self.positions.values()))
        self.bounds = geometry.bounds
        self.center = numpy.array([(self.bounds[0] + self.bounds[2])/2, (self.bounds[1] + self.bounds[3])/2, 0])

    def latlon_xy(self, latitude, longitude):
        """
        Transform latitude, longitude (GPS) coordinates to x, y (pixel) coordinates at specified zoom level,
          using Web Mercator projection.

        Based on https://stackoverflow.com/questions/62800466/transform-x-y-pixel-values-into-lat-and-long
        """
        c = (self.mts.get_tile_size(self.zoom_level) / (2 * math.pi)) * 2 ** self.zoom_level

        x = c * (math.radians(longitude) + math.pi)
        y = c * (math.pi - math.log(math.tan((math.pi / 4) + math.radians(latitude) / 2)))

        return x, y

    def transform(self, point: numpy.array):
        """Convert lat-lon coordinates into manim coordinates relative to self.center and self.zoom_level."""

        center_x, center_y = self.latlon_xy(self.center[0], self.center[1])
        point_x, point_y = self.latlon_xy(point[0], point[1])

        return numpy.array([(point_x - center_x) / manim.config.pixel_width * 14,
                            (point_y - center_y) / manim.config.pixel_height * -8, 0])

    def set_background(self):
        background_map = self.mts.get_map(self.center[0], self.center[1], self.zoom_level,
                                          manim.config.pixel_width, manim.config.pixel_height)

        print(f'PIXEL_HEIGHT {manim.config.pixel_height}')
        print(f'bg map height {background_map.height}')

        background_map = manim.ImageMobject(background_map, scale_to_resolution=manim.config.pixel_height)
        self.add(background_map)

    def construct(self):
        if defaults['use_map']:
            self.set_background()
        else:
            self.camera.background_color = defaults['background_color']

        targets = {}
        for node in self.rollout.data.targets:
            targets[node] = manim.Circle(fill_color=defaults['unprotected_fill'], radius=0.13, fill_opacity=0.8,
                                         color=manim.DARK_GRAY)
            target_position = list(self.rollout.data.target_geometry[node].centroid.coords)[0]
            targets[node].move_to(self.transform(target_position))
            targets[node].node_id = node
        target_creation = manim.Create(manim.VGroup(*targets.values()), run_time=1.0)

        locations = []
        number_of_units = len(self.rollout.data.base.factors)
        units = {}
        for unit in range(number_of_units):
            units[unit] = manim.Circle(color=manim.BLACK, radius=0.05, stroke_width=0.1, fill_opacity=0.9,
                                       fill_color=manim.DARK_BLUE)
            units[unit].move_to(self.transform(self.positions[self.rollout.schedule[0][unit]]))
            locations.append(self.positions[self.rollout.schedule[0][unit]])
        unit_creation = manim.Create(manim.VGroup(*units.values()), run_time=1.0)

        for node in self.positions:
            circle = manim.Circle(color=manim.DARK_GRAY, radius=0.005)
            circle.move_to(self.transform(self.positions[node]))
            self.add(circle)

        self.play(target_creation, unit_creation)

        unit_animation = {unit: [] for unit in range(number_of_units)}

        coverage = {target: {} for target in targets}
        for t, state in enumerate(self.rollout.schedule[1:]):
            for target in targets:
                coverage[target][(t+1)*self.speed] = self.rollout.data.base.coverage[state][target]

            destinations = []
            for unit, node in enumerate(state):
                destinations.append(self.positions[node])
                move_animation = manim.MoveAlongPath(units[unit],
                                                     manim.Line(self.transform(locations[unit]),
                                                                self.transform(destinations[unit])),
                                                     rate_func=defaults['rate']
                                                     )
                unit_animation[unit].append(move_animation)
            locations = destinations

        coverage_interpolation = {target: PWLMap(coverage[target]) for target in targets}

        def color_updater(mob, dt):
            target_node = mob.node_id
            mob.set_color(manim.interpolate_color(defaults['unprotected_border'], defaults['protected_border'],
                                                  coverage_interpolation[target_node](color_updater.tt[target_node])))
            mob.set_fill(manim.interpolate_color(defaults['unprotected_fill'],
                                                 defaults['protected_fill'],
                                                 coverage_interpolation[target_node](color_updater.tt[target_node])))
            color_updater.tt[target_node] += dt

        color_updater.tt = {target: 0 for target in targets}
        for target in targets:
            targets[target].add_updater(color_updater)

        unit_animation_groups = [manim.Succession(*unit_animation[unit],
                                                  run_time=(len(self.rollout.schedule) - 1) * self.speed)
                                 for unit in range(number_of_units)]

        self.play(
            *unit_animation_groups
        )
