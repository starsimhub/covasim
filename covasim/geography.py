"""
Geography: where agents live, which region and sewershed they are in.

Nothing in this file is specific to any one place. The caller supplies, as sim parameters:

- ``boundary_shapefile``: region polygons (GeoJSON or shapefile, with ``region_code``, ``region_name``, and optionally ``population``)
- ``density_raster``: a population-density raster (a GeoTIFF of people per pixel, in longitude/latitude, for example from WorldPop), and optionally ``density_bbox``, the window of it to read
- ``people_coords``: the location of each agent, as an (N, 2) array of longitude and latitude
- ``sewershed_file``: sewershed polygons (with a ``sid`` property); see sewersheds.py

``assign_geography()`` uses these to set the people's states ``x`` and ``y`` (their home
location, in degrees of longitude and latitude), ``region``, and ``sewershed``, when the
people are initialized (before the networks are made, so a network such as ``cv.RankNet``
can use the locations):

1. With a boundary file, people are placed inside its polygons: following the density raster,
   if given, with ``pop_type='rank'`` (each person's region is then the polygon they land in);
   otherwise, the regions are chosen in proportion to the ``population`` column (or the area),
   and people are placed uniformly within them.
2. Otherwise, the locations are ``people_coords``, if given; or, for ``pop_type='rank'``, drawn
   from the density raster, or (with no raster) spread uniformly over a 10 x 10 km square,
   which is only useful for testing.
3. With a sewershed file, each person's sewershed is the polygon they live in.

Region codes and sewershed IDs are stored as integers (see ``integer_ids()``); the original
codes are in ``sim.region_labels`` and ``sim.sewershed_labels``.

The raster library (rasterio) is an optional dependency, only needed by
load_density_raster(): pip install rasterio
"""

import numpy as np
import pandas as pd
from . import misc as cvm
from . import sewersheds as cvsew

__all__ = ['geo_pars', 'load_density_raster', 'sample_coords_from_density', 'sample_coords_within_boundaries', 'assign_regions_from_coords',
           'lonlat_to_km', 'lonlat_to_xyz_km', 'assign_density_based_coords', 'load_population_regions', 'assign_regions', 'assign_coords',
           'assign_geography']

# The sim parameters that say where agents live (see cv.make_pars())
geo_pars = ['boundary_shapefile', 'sewershed_file', 'people_coords', 'density_raster', 'density_bbox']

km_per_deg_lon = 111.320 # At the equator; multiply by cos(latitude) elsewhere
km_per_deg_lat = 110.574


def load_density_raster(path, bbox=None):
    """
    Read a population-count GeoTIFF (people per pixel) in longitude/latitude.

    Args:
        path (str): path to the .tif
        bbox (tuple): optional (lon_min, lat_min, lon_max, lat_max) window to read; None reads the whole file

    Returns:
        dict with 'counts' (2D array of people per pixel), 'lons' (1D pixel-centre longitudes),
        'lats' (1D pixel-centre latitudes), 'dlon', 'dlat' (pixel size in degrees)
    """
    try:
        import rasterio # Optional dependency
        from rasterio.windows import from_bounds
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = 'Reading a density raster needs the optional rasterio package: pip install rasterio'
        raise ModuleNotFoundError(errormsg) from E

    with rasterio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic: # The coordinate conversion below assumes degrees
            errormsg = f'The raster must use longitude/latitude (e.g. EPSG:4326), but its CRS is {src.crs}'
            raise ValueError(errormsg)
        if bbox is not None:
            window = from_bounds(*bbox, transform=src.transform)
            window = window.round_offsets().round_lengths()
        else:
            window = rasterio.windows.Window(0, 0, src.width, src.height)
        counts = src.read(1, window=window).astype(float)
        transform = src.window_transform(window)
        nodata = src.nodata

    if nodata is not None:
        counts[counts == nodata] = 0
    counts[~np.isfinite(counts)] = 0
    counts[counts < 0] = 0

    nrows, ncols = counts.shape
    dlon, dlat = transform.a, transform.e # dlat is negative for north-up rasters
    lons = transform.c + dlon*(np.arange(ncols) + 0.5)
    lats = transform.f + dlat*(np.arange(nrows) + 0.5)
    return dict(counts=counts, lons=lons, lats=lats, dlon=dlon, dlat=dlat)


def sample_coords_from_density(raster, pop_size):
    """
    Draw pop_size (lon, lat) points: pick a pixel with probability proportional
    to its population count, then place the agent uniformly within the pixel.

    Args:
        raster   (dict): as returned by load_density_raster()
        pop_size (int):  number of agents to place

    Returns:
        (pop_size, 2) array of longitude, latitude in degrees
    """
    counts = raster['counts']
    probs = counts.ravel()/counts.sum()
    pix = np.random.choice(probs.size, size=pop_size, p=probs)
    rows, cols = np.unravel_index(pix, counts.shape)
    lon = raster['lons'][cols] + (np.random.random(pop_size) - 0.5)*abs(raster['dlon'])
    lat = raster['lats'][rows] + (np.random.random(pop_size) - 0.5)*abs(raster['dlat'])
    return np.column_stack((lon, lat))


def sample_coords_within_boundaries(raster, boundaries, pop_size, max_tries=100):
    """
    Draw pop_size (lon, lat) points from a population-density raster, but only inside the boundary polygons.

    Pixels whose centre is outside every polygon get zero weight; the rest are chosen in proportion to
    their population, and the agent is placed at a random spot in the pixel. A spot that falls just outside
    the boundary (pixels on the edge) is redrawn until it is inside.

    Args:
        raster     (dict):         as returned by load_density_raster(), in longitude/latitude
        boundaries (GeoDataFrame): region polygons in longitude/latitude (EPSG:4326)
        pop_size   (int):          number of agents to place
        max_tries  (int):          maximum number of redraws for agents that land outside the boundary

    Returns:
        (pop_size, 2) array of longitude, latitude in degrees
    """
    import shapely # Imported here since only the boundary-based placement needs it

    boundary = shapely.union_all(boundaries.geometry.values) # All regions as one shape
    lon_grid, lat_grid = np.meshgrid(raster['lons'], raster['lats'])
    inside = shapely.contains_xy(boundary, lon_grid.ravel(), lat_grid.ravel()).reshape(lon_grid.shape)
    masked = dict(raster, counts=raster['counts']*inside) # People per pixel, zero outside the boundary
    if masked['counts'].sum() <= 0:
        errormsg = 'The density raster has no population inside the boundary polygons; check that they overlap and use the same coordinates (longitude/latitude)'
        raise ValueError(errormsg)

    pop_size = int(pop_size)
    coords = sample_coords_from_density(masked, pop_size)
    for _ in range(max_tries):
        outside = ~shapely.contains_xy(boundary, coords[:, 0], coords[:, 1])
        if not outside.any():
            return coords
        coords[outside] = sample_coords_from_density(masked, int(outside.sum())) # Redraw only those outside
    errormsg = f'Could not place all agents inside the boundary polygons after {max_tries} tries'
    raise RuntimeError(errormsg)


def assign_regions_from_coords(lonlat, boundaries):
    """
    Give each agent the region_code of the boundary polygon that contains its location.
    Points on a shared edge count as inside both polygons. Where polygons overlap, an agent in
    several polygons is assigned to one of them at random (equal probability) and a warning
    reports how many agents were affected, since regions are normally expected not to overlap.

    Args:
        lonlat     (array):        (N, 2) array of longitude, latitude in degrees
        boundaries (GeoDataFrame): region polygons (EPSG:4326) with a region_code column

    Returns:
        array of region_code values, one per agent
    """
    import shapely
    labels = boundaries['region_code'].to_numpy()
    lonlat = np.asarray(lonlat, dtype=float)

    # inside[i, j] is True if agent i is in polygon j
    inside = np.column_stack([shapely.intersects_xy(geom, lonlat[:, 0], lonlat[:, 1]) for geom in boundaries.geometry.values])
    n_matches = inside.sum(axis=1)
    if (n_matches == 0).any():
        errormsg = f'{int((n_matches == 0).sum())} agents are outside all boundary polygons'
        raise ValueError(errormsg)

    index = inside.argmax(axis=1) # The polygon each agent is in (for the usual case of exactly one)
    overlap = np.nonzero(n_matches > 1)[0]
    if len(overlap):
        # Pick uniformly among the matching polygons: give each a random score and take the highest of those that match
        scores = np.random.random(inside[overlap].shape)*inside[overlap]
        index[overlap] = scores.argmax(axis=1)
        warnmsg = f'{len(overlap)} of {len(lonlat)} agents are in more than one boundary polygon; each was assigned to one of them at random. Check the boundary file for overlapping regions.'
        cvm.warn(warnmsg)
    return labels[index]


def lonlat_to_km(lonlat):
    """
    Convert (lon, lat) in degrees to (x, y) in km, using an equirectangular projection
    about the mean latitude. This is accurate for areas of city to county size, so that
    straight-line distance between two rows is a good distance in km. It is not suitable
    for areas spanning many degrees of latitude or reaching the poles.

    Args:
        lonlat (array): (N, 2) array of longitude, latitude in degrees

    Returns:
        (N, 2) array of x, y in km, centred on the mean location
    """
    lonlat = np.asarray(lonlat, dtype=float)
    lon0, lat0 = lonlat[:, 0].mean(), lonlat[:, 1].mean()
    kx = km_per_deg_lon*np.cos(np.radians(lat0)) # km per degree of longitude at this latitude
    return np.column_stack(((lonlat[:, 0]-lon0)*kx, (lonlat[:, 1]-lat0)*km_per_deg_lat))


def lonlat_to_xyz_km(lonlat, radius=6371.0):
    """
    Convert (lon, lat) in degrees to 3D points (x, y, z) in km on a sphere the size of the Earth.

    The straight-line (chord) distance between two such points grows with the distance
    along the surface, so which points are nearest, second nearest, and so on is exactly the
    same as with great-circle distances, at any scale and anywhere on the globe. Use these
    as the coordinates for rank-based (nearest-neighbour) calculations. Chord lengths are
    very slightly shorter than surface distances (0.04% at 1,000 km); for rankings this
    makes no difference. The Earth is treated as a sphere, ignoring its slight flattening.

    Args:
        lonlat (array): (N, 2) array of longitude, latitude in degrees
        radius (float): sphere radius in km (default: mean Earth radius)

    Returns:
        (N, 3) array of x, y, z in km
    """
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(lonlat[:, 1])
    return radius*np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def assign_density_based_coords(pars):
    """
    Assign each agent a (longitude, latitude) location, sampled from a population-density raster.

    Args:
        pars (dict or Sim): needs 'pop_size'; optionally 'density_raster' (path to the .tif)
                            and 'density_bbox' (lon_min, lat_min, lon_max, lat_max)

    Returns:
        (N, 2) array of longitude and latitude in degrees. If no raster is given, agents are spread uniformly over
        a 10 x 10 km square at longitude and latitude 0, which is only useful for testing.

    | New in the v4 port: returns degrees rather than km, as the other ways of placing agents do (use lonlat_to_km() for km).
    """
    pop_size = int(pars['pop_size'])
    path = pars['density_raster'] if 'density_raster' in pars else None
    bbox = pars['density_bbox'] if 'density_bbox' in pars else None
    if path is None: # No raster: a 10 x 10 km test square
        km = np.random.uniform(0, 10, size=(pop_size, 2))
        return np.column_stack((km[:, 0]/km_per_deg_lon, km[:, 1]/km_per_deg_lat))
    raster = load_density_raster(path, bbox) # A grid of people per cell, plus the longitude/latitude of each cell
    return sample_coords_from_density(raster, pop_size) # A cell chosen in proportion to its population, then a random spot inside it


def load_population_regions(filename):
    """
    Load the region polygons of a boundary file (GeoJSON or shapefile, with columns region_code and region_name,
    and optionally population), with the probability of a person being in each region: in proportion to its
    population, or, with no population column, to its area.
    """
    import geopandas as gpd
    regions = gpd.read_file(filename)

    required = {'region_code', 'region_name'}
    missing = required - set(regions.columns)
    if missing:
        raise ValueError(f'Population boundary file is missing columns: {missing}')
    if regions.empty:
        raise ValueError('Population boundary file contains no regions.')

    if regions.crs is not None and not regions.crs.equals('EPSG:4326'): # Agent coordinates are longitude/latitude
        regions = regions.to_crs('EPSG:4326')

    if 'population' in regions.columns:
        regions['population'] = pd.to_numeric(regions['population'], errors='raise')
        if (regions['population'] < 0).any():
            raise ValueError('Population cannot be negative.')
        total_population = regions['population'].sum()
        if total_population <= 0:
            raise ValueError('Total population must be greater than zero.')
        regions['probability'] = regions['population'] / total_population
    else: # No population data: spread people evenly over the area, i.e. weight each region by its area
        cvm.warn('The boundary file has no "population" column, so regions are weighted by their area. '
                 'Add a "population" column to weight by population (it is not needed when pop_type="rank" uses a density_raster).')
        geometry = regions.geometry if regions.crs is not None else regions.geometry.set_crs('EPSG:4326')
        area = geometry.to_crs('EPSG:6933').area # An equal-area projection, so areas are comparable
        regions['probability'] = area / area.sum()

    regions['region_num'] = cvsew.integer_ids(regions['region_code'], name='region_code', minimum=None) # Integer ID used in people.region
    return regions


def assign_regions(pars, population_data):
    """ Choose a region (its region_code) for each of the pop_size agents, in proportion to the region probabilities (from load_population_regions()) """
    pop_size = int(pars['pop_size'])
    labels = population_data['region_code'].to_numpy()
    probabilities = population_data['probability'].to_numpy()
    return np.random.choice(labels, size=pop_size, p=probabilities)


def assign_coords(assigned_region_codes, population_data):
    """
    Generate coordinates for agents based on their assigned region geometry, uniformly within the polygon.

    Args:
        assigned_region_codes (array): the region_code of each agent
        population_data (GeoDataFrame): from load_population_regions()

    Returns:
        (N, 2) array of longitude and latitude
    """
    import shapely
    geoms = dict(zip(population_data['region_code'], population_data.geometry))
    codes = np.asarray(assigned_region_codes)
    coords = np.zeros((len(codes), 2))
    for code in np.unique(codes): # Rejection sampling inside each region's polygon, for all of its agents at once
        geom = geoms[code]
        inds = np.flatnonzero(codes == code)
        minx, miny, maxx, maxy = geom.bounds
        todo = inds
        while len(todo):
            x = np.random.uniform(minx, maxx, size=len(todo))
            y = np.random.uniform(miny, maxy, size=len(todo))
            inside = shapely.contains_xy(geom, x, y)
            coords[todo[inside], 0] = x[inside]
            coords[todo[inside], 1] = y[inside]
            todo = todo[~inside]
    return coords


def assign_geography(people, pars, pop_type=None):
    """
    Set the home location (``x`` and ``y``, in degrees of longitude and latitude), ``region``, and
    ``sewershed`` of each person, from the geography parameters (see the top of this file). Called when
    the people are initialized.

    Args:
        people (People): the people, with ``x``, ``y``, ``region``, and ``sewershed`` states
        pars (dict): the geography parameters (``geo_pars``), plus ``pop_size``
        pop_type (str): the population type; with 'rank', people are always given locations, since the network needs them

    Returns:
        Two dicts: {integer region ID: region_code} and {integer sewershed ID: original sid}
    """
    uids = people.auids
    n = len(uids)
    pars = dict(pars, pop_size=n)
    coords = pars.get('people_coords')
    region_labels = {}

    # Regions and locations from a boundary file
    boundary_file = pars.get('boundary_shapefile')
    if boundary_file is not None:
        if coords is not None:
            errormsg = 'Give either boundary_shapefile (to place people within it) or people_coords, not both'
            raise ValueError(errormsg)
        population_data = load_population_regions(boundary_file)
        if pop_type == 'rank' and pars.get('density_raster') is not None:
            # Place people according to the population-density raster, but only inside the boundary polygons;
            # each person's region is the polygon they land in
            bbox = pars['density_bbox'] if pars.get('density_bbox') is not None else tuple(population_data.total_bounds) # Read only the raster window around the boundaries
            raster = load_density_raster(pars['density_raster'], bbox)
            coords = sample_coords_within_boundaries(raster, population_data, n)
            regions = assign_regions_from_coords(coords, population_data)
        else: # Regions in proportion to the 'population' column, then uniform within each polygon
            regions = assign_regions(pars, population_data)
            coords = assign_coords(regions, population_data)
        region_nums = dict(zip(population_data['region_code'], population_data['region_num']))
        people.region[uids] = np.array([region_nums[code] for code in regions])
        region_labels = dict(zip(population_data['region_num'].tolist(), population_data['region_code'].tolist()))

    # Locations without a boundary file
    elif coords is None and pop_type == 'rank':
        coords = assign_density_based_coords(pars)

    # Store the locations
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (n, 2):
            errormsg = f'people_coords must have shape (pop_size, 2) = {(n, 2)}, not {coords.shape}'
            raise ValueError(errormsg)
        if not np.isfinite(coords).all():
            raise ValueError('people_coords must contain finite values')
        people.x[uids] = coords[:, 0]
        people.y[uids] = coords[:, 1]

    # Sewersheds
    sewershed_labels = cvsew.assign_sewersheds(people, pars.get('sewershed_file'))
    return region_labels, sewershed_labels
