"""
Generates some figures for UAI paper.
"""

import os
import warnings

import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, Stamen, OSM
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx
import numpy
import shapely

import stackelberg_games.patrolling as sgp


def main():
    gg = sgp.gdynia_graph()

    tiler = OSM(cache=True)
    web_mercator = tiler.crs

    positions = {}
    for node, position in networkx.get_node_attributes(gg, 'pos').items():
        point = numpy.array([position[0], position[1]])
        if node in positions and not numpy.allclose(positions[node], point):
            raise ValueError('Conflicting nodes in factor graphs.')
        positions[node] = point

    geometry = shapely.MultiPoint(list(positions.values()))
    bounds = shapely.buffer(geometry, 0.001).bounds
    center = numpy.array([(bounds[0] + bounds[2])/2, (bounds[1] + bounds[3])/2, 0])
    bbox = shapely.Polygon([[bounds[1], bounds[0]], [bounds[1], bounds[2]], [bounds[3], bounds[2]], [bounds[3], bounds[0]], [bounds[1], bounds[0]]])

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=web_mercator)
    print(f"Figure extent: {bounds}")
    ax.set_extent([bounds[1], bounds[3], bounds[0], bounds[2]], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 16)
    # ax.gridlines(draw_labels=True)

    for u, v in gg.to_undirected().edges:
        lat_u, lon_u = positions[u]
        lat_v, lon_v = positions[v]

        print(f'Plotting line ({lon_u:.6f}, {lat_u:.6f}) -- ({lon_v:.6f}, {lat_v:.6f})')
        ax.plot([lon_u, lon_v], [lat_u, lat_v], '--',
                color='black', linewidth=1, alpha=0.5,
                transform=ccrs.PlateCarree(), zorder=4)

    rect_w = 0.001
    rect_h = 0.0005
    for node, position in positions.items():
        if node.startswith('B_'): # Docks - attack targets
            print(f"Plotting square at ({position[1]:.6f}, {position[0]:.6f})")
            ax.add_patch(mpatches.Rectangle(xy=[position[1]-rect_w/2, position[0]-rect_h/2], width=rect_w, height=rect_h, facecolor='red', edgecolor='black', alpha=0.5, transform=ccrs.PlateCarree()))
        else:
            print(f"Plotting circle at ({position[1]:.6f}, {position[0]:.6f})")
            ax.tissot(rad_km=0.02, lons=[position[1]], lats=[position[0]], transform=ccrs.PlateCarree(), alpha=0.5, edgecolor='black', zorder=5)

    os.makedirs(sgp.directories.plots_dir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Approximating coordinate system.*")
        plt.savefig(f'{sgp.directories.plots_dir}/gdynia_diagram.png', bbox_inches='tight')
    print(f"\nSaved diagram to {f'{sgp.directories.plots_dir}/gdynia_diagram.png'}")

if __name__ == '__main__':
    main()
