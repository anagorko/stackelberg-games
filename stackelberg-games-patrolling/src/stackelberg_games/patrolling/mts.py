"""
This module provides access to map tile services.

Input:
    latitude, longitude - coordinates of the center point
    zoom                - map zoom level
    width, height       - map extent in pixels

Output:
    a raster image of the map
"""

import math
import json
import io
from pathlib import Path
from urllib.request import urlopen, Request
from PIL import Image
from tqdm import tqdm

import directories


class MapTileService:
    """
    Access to map tile services.

    Attributes:
        url - map tile provider URL.

    TODO: tile caching on disk, attribution/copyrights
    https://realpython.com/storing-images-in-python/
    """

    map_tile_provider = {
        'osm': {
            'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            'attribution': ''
        },
        'google_terrain': {
            'url': 'http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}&s=Ga',
            'attribution': ''
        },
        'stadia_dark': {
            'url': 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
            'attribution': '',
            'credentials': 'stadia_maps_credentials.json'
        },
        'stadia_smooth': {
            'url': 'https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png',
            'attribution': '',
            'credentials': 'stadia_maps_credentials.json'
        },
        'toner-lite': {
            'url': 'https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png',
            'attribution': '',
            'credentials': 'stadia_maps_credentials.json'
        },
        'toner': {
            'url': 'https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}{r}.png',
            'attribution': '',
            'credentials': 'stadia_maps_credentials.json'
        }
    }

    def __init__(self, service='osm'):
        if service not in self.map_tile_provider:
            raise ValueError(f'Unknown tile provider "{service}"')

        self.url = self.map_tile_provider[service]['url']
        self.attribution = self.map_tile_provider[service]['attribution']

        if 'credentials' in self.map_tile_provider[service]:
            try:
                file = open(self.map_tile_provider[service]['credentials'], 'r')
            except FileNotFoundError:
                raise ValueError(f'You need to provide a credentials file "'
                                 f'{self.map_tile_provider[service]["credentials"]}"')
            self.credentials = json.load(file)
            file.close()
        else:
            self.credentials = None

        self.service = service
        self.cache = {}
        self._tile_size = {}
        self.disk_cache_dir = Path(directories.tile_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _tile_url(self, x, y, z):
        """Return map tile URL for specified coordinates."""
        return self.url.format(x=x, y=y, z=z, r='')

    def retrieve_tile(self, x, y, z, progress_bar=None):
        """
        Retrieve tile from map tile service or from cache.

        Based on
          https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
          https://www.theurbanist.com.au/2021/03/plotting-openstreetmap-images-with-cartopy/
        """

        disk_cache_filename = f'{self.service}_{x}_{y}_{z}.png'

        if (x, y, z) not in self.cache:
            try:
                self.cache[x, y, z] = Image.open(self.disk_cache_dir / disk_cache_filename)
            except FileNotFoundError:
                if progress_bar:
                    progress_bar.set_description(f'Cached tile {x}_{y}_{z} not found')
                else:
                    print(f'Cached tile {x}_{y}_{z} not found.')

        if (x, y, z) not in self.cache:
            url = self._tile_url(x, y, z)
            req = Request(url)
            req.add_header('User-agent', 'Security Games/0.1')
            if self.credentials:
                for key in self.credentials:
                    req.add_header(key, self.credentials[key])
            fh = urlopen(req)
            im_data = io.BytesIO(fh.read())
            fh.close()
            self.cache[x, y, z] = Image.open(im_data).convert('RGBA')
            self.cache[x, y, z].save(self.disk_cache_dir / disk_cache_filename)

        return self.cache[x, y, z]

    def merge_tiles(self, tiles):
        """
        Paste tiles onto a single image.
        """

        xyz_with_max_z = max(tiles, key=lambda xyz: xyz[2])
        zoom_level = xyz_with_max_z[2]
        tile_size = self.get_tile_size(zoom_level)

        west, east, north, south = tile_size * 2 ** zoom_level, 0, tile_size * 2 ** zoom_level, 0
        for x, y, z in tiles:
            west = min(west, x * tile_size * 2 ** (zoom_level - z))
            east = max(east, (x + 1) * tile_size * 2 ** (zoom_level - z))
            south = max(south, (y + 1) * tile_size * 2 ** (zoom_level - z))
            north = min(north, y * tile_size * 2 ** (zoom_level - z))

        width = east - west
        height = south - north
        if width > 2 ** 12 or height > 2 ** 11:
            raise ValueError(f'merge_tiles: refusing to create image with dimensions {width}x{height}.')

        print(f'width {width}, height {height}')
        image = Image.new('RGBA', (width, height))

        progress_bar = tqdm(sorted(tiles, key=lambda xyz: xyz[2]), desc='Fetching tiles')
        for x, y, z in progress_bar:
            x_px = x * tile_size * 2 ** (zoom_level - z) - west
            y_px = y * tile_size * 2 ** (zoom_level - z) - north

            tile = self.retrieve_tile(x, y, z, progress_bar)
            if zoom_level - z > 0:
                tile = tile.resize((tile_size * 2 ** (zoom_level - z), tile_size * 2 ** (zoom_level - z)),
                                   resample=Image.LANCZOS)
            image.paste(tile, (x_px, y_px))

        return image

    def get_map(self, latitude, longitude, zoom_level, width, height):
        """
        Return map image centered at lat-lon at given zoom level with specified width and height.
        """

        tile_size = self.get_tile_size(zoom_level)
        c_x, c_y = self.projection(latitude, longitude, zoom_level)

        c_x = int(c_x)
        c_y = int(c_y)

        west = c_x - width / 2
        east = c_x + width / 2
        north = c_y - height / 2
        south = c_y + height / 2

        x_tile_range = range(math.floor(west / tile_size), math.floor(east / tile_size) + 1)
        y_tile_range = range(math.floor(north / tile_size), math.floor(south / tile_size) + 1)

        tiles = {(x, y, zoom_level) for x in x_tile_range for y in y_tile_range}

        merged_tiles = self.merge_tiles(tiles)

        return merged_tiles.crop((
            west - math.floor(west / tile_size) * tile_size,
            north - math.floor(north / tile_size) * tile_size,
            west - math.floor(west / tile_size) * tile_size + width,
            north - math.floor(north / tile_size) * tile_size + height
        ))

    def get_tile_size(self, zoom_level):
        if zoom_level not in self._tile_size:
            tile = self.retrieve_tile(0, 0, zoom_level)
            assert tile.width == tile.height
            self._tile_size[zoom_level] = tile.width

        return self._tile_size[zoom_level]

    def projection(self, latitude, longitude, zoom_level):
        """
        Transform latitude, longitude (GPS) coordinates to x, y (pixel) coordinates at specified zoom level,
          using Web Mercator projection.

        Based on https://stackoverflow.com/questions/62800466/transform-x-y-pixel-values-into-lat-and-long
        """
        C = (self.get_tile_size(zoom_level) / (2 * math.pi)) * 2 ** zoom_level

        x = C * (math.radians(longitude) + math.pi)
        y = C * (math.pi - math.log(math.tan((math.pi / 4) + math.radians(latitude) / 2)))

        return x, y
