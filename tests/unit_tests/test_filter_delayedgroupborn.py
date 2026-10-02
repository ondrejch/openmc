import h5py
import numpy as np
import openmc
import openmc.lib
import pytest


def test_delayed_group_born_filter():
    f = openmc.DelayedGroupBornFilter([0, 1, 2, 3, 4, 5, 6])
    assert f.num_bins == 7
    assert f.shape == (7,)
    assert list(f.bins) == list(range(7))

    # Make sure __repr__ works
    repr(f)

    # Equality depends on the type and the bins
    assert f == openmc.DelayedGroupBornFilter(list(range(7)))
    assert f != openmc.DelayedGroupBornFilter([0, 1])
    assert (openmc.DelayedGroupBornFilter([1, 2]) !=
            openmc.DelayedGroupFilter([1, 2]))

    # to_xml_element()
    elem = f.to_xml_element()
    assert elem.tag == 'filter'
    assert elem.attrib['type'] == 'delayedgroupborn'
    assert elem.attrib['id'] == str(f.id)
    assert elem.find('bins').text == '0 1 2 3 4 5 6'

    # from_xml_element()
    new_f = openmc.Filter.from_xml_element(elem)
    assert isinstance(new_f, openmc.DelayedGroupBornFilter)
    assert new_f.id == f.id
    assert list(new_f.bins) == list(f.bins)


def test_delayed_group_born_filter_invalid_bins():
    # Group 0 (born prompt) and the maximum number of delayed groups are valid
    max_groups = openmc.mgxs.MAX_DELAYED_GROUPS
    assert openmc.DelayedGroupBornFilter([0, max_groups]).num_bins == 2

    with pytest.raises(ValueError):
        openmc.DelayedGroupBornFilter([-1, 0])
    with pytest.raises(ValueError):
        openmc.DelayedGroupBornFilter([max_groups + 1])
    with pytest.raises(ValueError):
        openmc.DelayedGroupBornFilter([0, 1, 1])
    with pytest.raises(TypeError):
        openmc.DelayedGroupBornFilter([0, 1.5])

    # An invalid assignment leaves the bins unchanged
    f = openmc.DelayedGroupBornFilter([0, 1])
    with pytest.raises(ValueError):
        f.bins = [2, 2]
    assert f.bins == [0, 1]


def test_delayed_group_born_filter_from_hdf5(run_in_tmpdir):
    # Mimic the group that is written to a statepoint for this filter
    with h5py.File('filter.h5', 'w') as fh:
        group = fh.create_group('tallies/filters/filter 17')
        group.create_dataset('type', data=np.bytes_('delayedgroupborn'))
        group.create_dataset('n_bins', data=3)
        group.create_dataset('bins', data=np.array([0, 2, 5], dtype=np.int32))

    with h5py.File('filter.h5', 'r') as fh:
        f = openmc.Filter.from_hdf5(fh['tallies/filters/filter 17'])
    assert isinstance(f, openmc.DelayedGroupBornFilter)
    assert f.id == 17
    assert f.num_bins == 3
    assert list(f.bins) == [0, 2, 5]


def test_delayed_group_born_filter_dataframe():
    f = openmc.DelayedGroupBornFilter([0, 3])

    # Filter with a stride of two (one more filter with two bins after it) in a
    # tally with another filter of two bins before it
    df = f.get_pandas_dataframe(8, 2)
    assert list(df.columns) == ['delayedgroupborn']
    assert list(df['delayedgroupborn']) == [0, 0, 3, 3] * 2


def test_delayed_group_born_filter_lib(run_in_tmpdir):
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xsdata = openmc.XSdata('medium', groups)
    xsdata.order = 0
    xsdata.set_total([1.0])
    xsdata.set_absorption([0.25])
    xsdata.set_scatter_matrix([[[0.75]]])
    library = openmc.MGXSLibrary(groups)
    library.add_xsdata(xsdata)
    library.export_to_hdf5('one_group.h5')

    medium = openmc.Material()
    medium.set_density('macro', 1.0)
    medium.add_macroscopic('medium')

    model = openmc.Model()
    model.materials = openmc.Materials([medium])
    model.materials.cross_sections = 'one_group.h5'
    box = openmc.model.RectangularParallelepiped(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, boundary_type='reflective')
    model.geometry = openmc.Geometry([openmc.Cell(fill=medium, region=-box)])
    model.settings.energy_mode = 'multi-group'
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 2
    model.settings.particles = 100
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Discrete([1.0e6], [1.0]))

    tally = openmc.Tally(tally_id=7)
    tally.filters = [openmc.DelayedGroupBornFilter([0, 1, 4], filter_id=11)]
    tally.scores = ['absorption']
    model.tallies = [tally]

    model.init_lib(output=False)
    try:
        # The filter read from XML is mapped to its openmc.lib class, and the
        # default estimator is kept
        lib_tally = openmc.lib.tallies[7]
        assert lib_tally.estimator == 'tracklength'
        lib_filter = lib_tally.filters[0]
        assert isinstance(lib_filter, openmc.lib.DelayedGroupBornFilter)
        assert lib_filter.id == 11
        assert lib_filter.n_bins == 3
        assert isinstance(openmc.lib.filters[11],
                          openmc.lib.DelayedGroupBornFilter)

        # A filter of this type can be created through the C API
        new_filter = openmc.lib.DelayedGroupBornFilter()
        assert new_filter.filter_type == 'delayedgroupborn'
        assert isinstance(openmc.lib.filters[new_filter.id],
                          openmc.lib.DelayedGroupBornFilter)

        # External source particles are born prompt
        openmc.lib.run(output=False)
        mean = lib_tally.mean.ravel()
        assert mean[0] > 0.0
        assert np.all(mean[1:] == 0.0)
    finally:
        model.finalize_lib()
