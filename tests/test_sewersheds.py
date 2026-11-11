"""
Test sewershed ID handling and assignment. Needs geopandas (a covasim dependency) but no external files.
"""
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box
import covasim as cv
from covasim.sewersheds import sewershed_id_map


def make_file(tmp_path, sids):
    """ Write side-by-side unit squares, one per sid: square i covers x in [i, i+1], y in [0, 1] """
    gdf = gpd.GeoDataFrame({'sid': sids}, geometry=[box(i, 0, i+1, 1) for i in range(len(sids))], crs='EPSG:4326')
    path = str(tmp_path/'sewersheds.geojson')
    gdf.to_file(path, driver='GeoJSON')
    return path


def run_sim(path, n=300):
    """ People at the centre of each square (cycling), plus a few outside all squares """
    n_sheds = len(sewershed_id_map(path))
    x = (np.arange(n) % n_sheds) + 0.5
    y = np.full(n, 0.5)
    x[-5:] = 100 # Outside every polygon
    sim = cv.Sim(pop_size=n, n_days=2, verbose=0, sewershed_file=path, people_coords=np.column_stack((x, y)))
    sim.initialize()
    return sim


def test_labelled_sids_numbered_in_file_order(tmp_path):
    """ Non-integer sids get 1..n in file order; "A1" and "B1" must not collide """
    path = make_file(tmp_path, ['A1', 'B1', 'plant_C'])
    assert sewershed_id_map(path) == {1: 'A1', 2: 'B1', 3: 'plant_C'}
    assert cv.sewershed_id_map is sewershed_id_map # Exported flat
    sim = run_sim(path)
    assert sim.sewershed_labels == {1: 'A1', 2: 'B1', 3: 'plant_C'} # Stored on the sim
    sh = sim.people.sewershed.values
    assert set(np.unique(sh)) == {-1, 1, 2, 3}
    assert (sh[:3] == [1, 2, 3]).all()
    assert (sh[-5:] == -1).all() # Outside all polygons is distinct from every sewershed


def test_no_sewershed_file_has_no_labels():
    """ Without a sewershed file everyone is in catchment 0 and there are no labels """
    sim = cv.Sim(pop_size=200, n_days=2, verbose=0).initialize()
    assert sim.sewershed_labels == {}
    assert (sim.people.sewershed == 0).all()


def test_integer_sids_kept(tmp_path):
    """ Integer sids (as numbers or strings) are used as given, not renumbered """
    for sids in ([5, 9], ['5', '9']):
        path = make_file(tmp_path, sids)
        assert sewershed_id_map(path) == {5: sids[0], 9: sids[1]}
        sh = run_sim(path).people.sewershed.values
        assert set(np.unique(sh)) == {-1, 5, 9}


def test_reserved_integer_sids_rejected(tmp_path):
    """ 0 and -1 are reserved for the default catchment and for outside all polygons """
    for sids in ([0, 1], [-1, 1]):
        with pytest.raises(ValueError):
            sewershed_id_map(make_file(tmp_path, sids))


if __name__ == '__main__':
    import pathlib, tempfile
    for test in [test_labelled_sids_numbered_in_file_order, test_integer_sids_kept, test_reserved_integer_sids_rejected]:
        test(pathlib.Path(tempfile.mkdtemp()))
    test_no_sewershed_file_has_no_labels()
