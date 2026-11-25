"""Convert a projected bare-earth GeoTIFF into a Blender-ready OBJ.
Install dependencies: python3 -m pip install numpy rasterio
Run: python3 terrain_transfer.py
"""
from pathlib import Path
import json
import numpy as np
import rasterio

# ================= EDIT THESE SETTINGS =================
FOLDER = Path(__file__).resolve().parent
INPUT_FILE = FOLDER / 'nj2015_del_valley_dem_J1495128tR0_C0.tif'
OUTPUT_FILE = FOLDER / 'campus_terrain.obj'
GRID_SPACING = 2.0  # meters; use 1.0 for native detail
VERTICAL_EXAGGERATION = 1.0  # 1 preserves real slopes
NORMALIZE_ELEVATION = True  # lowest valid source elevation becomes zero
CENTER_TERRAIN = True  # center horizontal coordinates around (0, 0)
# Override only if the raster's vertical units are missing or incorrect.
VERTICAL_METERS_PER_UNIT = None
# =======================================================


def convert_terrain():
    if GRID_SPACING <= 0 or VERTICAL_EXAGGERATION <= 0:
        raise ValueError('Spacing and exaggeration must be positive.')
    with rasterio.open(INPUT_FILE) as source:
        if not source.crs or not source.crs.is_projected:
            raise ValueError('A projected raster CRS is required.')
        horizontal_factor = source.crs.linear_units_factor[1]
        units = (source.units[0] or '').lower().strip()
        unit_factors = {'us survey foot': 1200/3937, 'us survey feet': 1200/3937,
                        'foot': 0.3048, 'feet': 0.3048, 'ft': 0.3048,
                        'meter': 1, 'meters': 1, 'metre': 1, 'metres': 1, 'm': 1}
        vertical_factor = VERTICAL_METERS_PER_UNIT
        if vertical_factor is None:
            vertical_factor = unit_factors.get(units)
        if vertical_factor is None or vertical_factor <= 0:
            raise ValueError('Set VERTICAL_METERS_PER_UNIT using dataset metadata.')
        transform = source.transform
        crs = source.crs.to_string()
        # Read elevations, respecting the raster's NoData mask.
        elevation = source.read(1, masked=True).astype(np.float64).filled(np.nan)
        elevation = (elevation * source.scales[0] + source.offsets[0]) * vertical_factor
        valid = np.isfinite(elevation)
        if not valid.any():
            raise ValueError('No valid elevation samples.')
        minimum = float(elevation[valid].min())
        maximum = float(elevation[valid].max())
        height, width = elevation.shape
        origin_x, origin_y = transform * (width / 2, height / 2) if CENTER_TERRAIN else (0, 0)
        origin_z = minimum if NORMALIZE_ELEVATION else 0.0
        # Sample the native grid without inventing extra detail.
        pixel_x = np.hypot(transform.a, transform.d) * horizontal_factor
        pixel_y = np.hypot(transform.b, transform.e) * horizontal_factor
        step_x = max(1, round(GRID_SPACING / pixel_x))
        step_y = max(1, round(GRID_SPACING / pixel_y))
        rows = np.arange(0, height, step_y)
        cols = np.arange(0, width, step_x)
        sampled = elevation[np.ix_(rows, cols)]
        rr, cc = np.meshgrid(rows + 0.5, cols + 0.5, indexing='ij')
        # Pixel centers become Blender X (east), Y (north), Z (up).
        x = (transform.a * cc + transform.b * rr + transform.c - origin_x) * horizontal_factor
        y = (transform.d * cc + transform.e * rr + transform.f - origin_y) * horizontal_factor
        z = (sampled - origin_z) * VERTICAL_EXAGGERATION
        # Reject coarse cells that cross any missing source data.
        missing = (~valid).astype(np.int64)
        integral = np.pad(missing, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        r0, r1 = rows[:-1, None], rows[1:, None] + 1
        c0, c1 = cols[None, :-1], cols[None, 1:] + 1
        good_cells = (integral[r1,c1]-integral[r0,c1]-integral[r1,c0]+integral[r0,c0]) == 0
        # Only export vertices belonging to at least one complete cell.
        used = np.zeros(sampled.shape, dtype=bool)
        used[:-1,:-1] |= good_cells
        used[1:,:-1] |= good_cells
        used[:-1,1:] |= good_cells
        used[1:,1:] |= good_cells
        count = int(used.sum())
        if not count:
            raise ValueError('No complete mesh cells; try a smaller GRID_SPACING.')
        indices = np.zeros(sampled.shape, dtype=np.int64)
        indices[used] = np.arange(1, count + 1)  # OBJ indices start at 1
        # Ensure faces point upward even if raster axes are reversed.
        reverse = transform.a * transform.e - transform.b * transform.d > 0
        with OUTPUT_FILE.open('w') as obj:
            obj.write('# Bare-earth terrain; meters; Z up, Y north\no Campus_Terrain\n')
            np.savetxt(obj, np.column_stack((x[used], y[used], z[used])), fmt='v %.5f %.5f %.5f')
            for r, c in zip(*np.nonzero(good_cells)):
                a,b,d,e = indices[r,c],indices[r,c+1],indices[r+1,c],indices[r+1,c+1]
                for face in ((a,d,b),(b,d,e)):
                    if reverse:
                        face = face[::-1]
                    obj.write('f %d %d %d\n' % face)
        # Save the real-world origin for future buildings or GIS alignment.
        metadata = dict(source_crs=crs, source_origin_xy=[origin_x,origin_y],
                        horizontal_meters_per_unit=horizontal_factor,
                        vertical_meters_per_unit=vertical_factor,
                        elevation_origin_m=origin_z, elevation_range_m=[minimum,maximum],
                        grid_spacing_m=[pixel_x*step_x,pixel_y*step_y],
                        vertical_exaggeration=VERTICAL_EXAGGERATION,
                        vertices=count, triangles=int(good_cells.sum())*2)
        OUTPUT_FILE.with_suffix('.json').write_text(json.dumps(metadata, indent=4))
        print(json.dumps(metadata, indent=4))
        print(f'Saved {OUTPUT_FILE}')


if __name__ == '__main__':
    convert_terrain()
