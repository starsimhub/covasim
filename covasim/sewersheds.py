"""
Sewersheds: which wastewater catchment each person lives in.

A sewershed file is a GeoJSON or shapefile of polygons, each with a unique ``sid``
property. Each person's sewershed (``people.sewershed``) is the integer ID of the
polygon that contains their home location (``people.x``, ``people.y``, in degrees of
longitude and latitude); 0 means there is no sewershed file (one catchment for
everyone), and -1 means outside all of the polygons. ``sim.sewershed_labels`` maps the
integer IDs back to the original sids.
"""

import numpy as np
import pandas as pd

__all__ = ['sewershed_id_map', 'load_sewersheds', 'assign_sewersheds']


def integer_ids(values, name='sid', minimum=1):
    """
    Turn a column of codes (e.g. the 'sid' property) into integer IDs, which is what the people's states store.

    If every value is already an integer (1, 2, ... or "1", "2", "01", ...) those values are used as they are.
    Otherwise (e.g. "SWR-01", "plant_A") the values are numbered 1, 2, 3, ... in file order.
    For sewersheds, 0 (default catchment) and -1 (outside all polygons) are reserved, so IDs must be >= 1.

    Args:
        values (Series): the codes
        name (str): the name of the column, for error messages
        minimum (int): the smallest integer ID allowed, or None for no minimum
    """
    as_str = values.astype(str).str.strip()
    is_integer = as_str.str.fullmatch(r'-?\d+').all()
    if not is_integer and pd.api.types.is_float_dtype(values): # e.g. 1.0, 2.0 from a numeric GeoJSON property
        is_integer = bool((values % 1 == 0).all())
        as_str = values.astype(int).astype(str)
    if is_integer:
        ids = as_str.astype(int).to_numpy()
        if minimum is not None and (ids < minimum).any():
            errormsg = f'Integer {name} values must be >= {minimum} (0 and -1 are reserved for the default catchment and for outside all polygons).'
            raise ValueError(errormsg)
    else:
        ids = np.arange(1, len(values) + 1)
    if len(np.unique(ids)) != len(ids): # Cannot normally happen, since the original codes are unique
        raise ValueError(f'{name} values must be unique.')
    return ids


def sewershed_id_map(filename):
    """
    Return {integer ID: original sid} for a sewershed file, i.e. the number stored in
    people.sewershed (and passed as sid= to WastewaterSampler) for each sewershed.
    """
    gdf = load_sewersheds(filename)
    return dict(zip(gdf['sid_num'].tolist(), gdf['sid'].tolist()))


def load_sewersheds(filename):
    """Load sewershed polygons from a GeoJSON/shapefile, with their integer IDs (column 'sid_num')."""
    import geopandas as gpd
    gdf = gpd.read_file(filename)

    if gdf.empty:
        raise ValueError('Sewershed file contains no polygons.')
    if 'sid' not in gdf.columns:
        raise ValueError("Sewershed GeoJSON must contain a 'sid' property.")
    if gdf['sid'].isna().any():
        raise ValueError('Every sewershed must have a sid.')
    if gdf['sid'].duplicated().any():
        raise ValueError('Sewershed sid values must be unique.')
    if not gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('Sewershed geometries must be Polygon or MultiPolygon.')
    if not gdf.geometry.is_valid.all():
        raise ValueError('Sewershed GeoJSON contains invalid geometries.')

    gdf['sid_num'] = integer_ids(gdf['sid']) # Integer ID used in people.sewershed
    return gdf


def assign_sewersheds(people, filename=None):
    """
    Set people.sewershed from their home locations (people.x and people.y, in longitude and latitude).

    Args:
        people (People): the people
        filename (str): the sewershed file; if None, everyone is in catchment 0

    Returns:
        {integer ID: original sid}
    """
    uids = people.auids
    if filename is None:
        people.sewershed[uids] = 0
        return {}

    x = people.x[uids]
    y = people.y[uids]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        errormsg = 'Sewershed assignment requires a location for each person: give boundary_shapefile or people_coords (or use pop_type="rank")'
        raise ValueError(errormsg)

    import geopandas as gpd
    sewersheds = load_sewersheds(filename)
    if sewersheds.crs is None: # Assume longitude/latitude if the file doesn't say
        sewersheds = sewersheds.set_crs('EPSG:4326')
    elif not sewersheds.crs.equals('EPSG:4326'): # People's locations are in longitude/latitude
        sewersheds = sewersheds.to_crs('EPSG:4326')

    # Find the polygon each person is in, with a spatial join
    points = gpd.GeoDataFrame({'agent': np.arange(len(uids))}, geometry=gpd.points_from_xy(x, y), crs=sewersheds.crs)
    joined = gpd.sjoin(points, sewersheds[['sid_num', 'geometry']], how='left', predicate='intersects')
    joined = joined.drop_duplicates(subset=['agent'], keep='first') # A person on the edge of two polygons is in the first

    assignments = np.full(len(uids), -1, dtype=int) # -1: outside all polygons
    assignments[joined['agent'].to_numpy()] = joined['sid_num'].fillna(-1).to_numpy(dtype=int)
    people.sewershed[uids] = assignments
    return dict(zip(sewersheds['sid_num'].tolist(), sewersheds['sid'].tolist())) # {integer ID: original sid}
