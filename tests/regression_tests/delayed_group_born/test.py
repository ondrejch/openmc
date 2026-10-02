"""Test of the delayed group born filter in an infinite multigroup medium.

The medium has four energy groups and three delayed groups, with the same
delayed neutron fractions beta_d in every energy group. Prompt fission
neutrons are emitted in the first energy group and the delayed neutrons of
delayed group d in energy group d + 1, and scattering never changes the energy
group. A neutron therefore stays in the energy group in which it was born, so
every event in energy group g is caused by a neutron born from delayed group g
(with group 0 meaning born prompt). The tally by birth delayed group and
energy group must be diagonal, and its diagonal must equal the tally by energy
group alone, for any estimator. This holds for any random number sequence. It
fails if the filter used Particle::delayed_group(), which multigroup fission
site creation overwrites with the delayed group of the last fission site that
the particle creates: a neutron born prompt that has created a delayed site
would then score off the diagonal.

The fraction of the neutrons born from each delayed group is also compared
with its expected value to within four standard deviations, using an analog
absorption tally: every neutron is absorbed exactly once with unit weight.

* In an eigenvalue calculation, the source neutrons are fission neutrons, and
  each fission neutron is born from delayed group d with probability beta_d,
  whatever the energy group of the neutron that caused the fission. The
  fractions of source neutrons born prompt and from delayed group d are thus
  1 - beta and beta_d, with beta = sum_d beta_d, and the total absorption is
  one per source neutron.
* In a fixed-source calculation with fission neutrons, an external source
  neutron in the first energy group (born prompt, group 0) produces on
  average F = k_0 / (1 - k) fission neutrons in all generations, where
  k_g = nu Sigma_f,g / Sigma_a,g is the number of fission neutrons produced
  by a neutron of energy group g and k = (1 - beta) k_0 + sum_d beta_d k_d is
  the mean over the birth spectrum (also the eigenvalue). Per source neutron,
  1 + (1 - beta) F neutrons are born prompt and beta_d F from delayed
  group d.

Finally, the bins of the filter add up to the tally without it, and a delayed
group filter combined with the delayed group born filter gives the delayed
production of each precursor group, beta_d times the production, for every
birth group.

"""
import numpy as np
import openmc
import pytest

from tests.regression_tests import config

# Some estimates are exact in this problem, and the sample standard deviation
# of a constant can evaluate to the square root of a slightly negative number
pytestmark = pytest.mark.filterwarnings(
    'ignore:invalid value encountered in sqrt:RuntimeWarning')

# Four-group data with three delayed groups; group 0 is the highest energy
ENERGY_EDGES = [0.0, 1.0, 1.0e2, 1.0e4, 20.0e6]
SIGMA_T = np.array([1.0, 1.2, 1.5, 2.0])  # 1/cm
SIGMA_A = np.array([0.2, 0.3, 0.4, 0.5])  # 1/cm
NU_SIGMA_F = np.array([0.12, 0.15, 0.2, 0.25])  # 1/cm
BETA = np.array([0.04, 0.08, 0.12])
DECAY_RATE = np.array([0.1, 1.0, 10.0])  # 1/s
N_GROUPS = SIGMA_T.size
N_DELAYED = BETA.size

# Birth delayed groups 0 (prompt), 1, ..., N_DELAYED. A neutron born from
# delayed group j is in energy group j, which is bin N_GROUPS - 1 - j of an
# energy filter with increasing energies.
BORN_GROUPS = list(range(N_DELAYED + 1))
ENERGY_BIN = [N_GROUPS - 1 - j for j in BORN_GROUPS]

# Expected fractions of the neutrons born prompt and from each delayed group
BIRTH_SPECTRUM = np.concatenate(([1.0 - BETA.sum()], BETA))
K_GROUP = NU_SIGMA_F / SIGMA_A
K_INF = BIRTH_SPECTRUM @ K_GROUP
FISSION_NEUTRONS = K_GROUP[0] / (1.0 - K_INF)


def make_model(library):
    groups = openmc.mgxs.EnergyGroups(ENERGY_EDGES)
    xsdata = openmc.XSdata('medium', groups, num_delayed_groups=N_DELAYED)
    xsdata.order = 0
    xsdata.set_total(SIGMA_T)
    xsdata.set_absorption(SIGMA_A)
    xsdata.set_scatter_matrix(np.diag(SIGMA_T - SIGMA_A)[:, :, np.newaxis])
    xsdata.set_prompt_nu_fission((1.0 - BETA.sum()) * NU_SIGMA_F)
    xsdata.set_delayed_nu_fission(np.outer(BETA, NU_SIGMA_F))
    xsdata.set_chi_prompt(np.eye(N_GROUPS)[0])
    xsdata.set_chi_delayed(np.eye(N_GROUPS)[1:N_DELAYED + 1])
    xsdata.set_decay_rate(DECAY_RATE)
    mgxs_library = openmc.MGXSLibrary(groups, num_delayed_groups=N_DELAYED)
    mgxs_library.add_xsdata(xsdata)
    mgxs_library.export_to_hdf5(library)

    medium = openmc.Material()
    medium.set_density('macro', 1.0)
    medium.add_macroscopic('medium')

    model = openmc.Model()
    model.materials = openmc.Materials([medium])
    model.materials.cross_sections = str(library)
    box = openmc.model.RectangularParallelepiped(
        -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, boundary_type='reflective')
    model.geometry = openmc.Geometry([openmc.Cell(fill=medium, region=-box)])

    model.settings.energy_mode = 'multi-group'
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Discrete([1.0e6], [1.0]))

    # Tallies by birth delayed group, each with a reference tally without the
    # delayed group born filter
    born = openmc.DelayedGroupBornFilter(BORN_GROUPS)
    energy = openmc.EnergyFilter(ENERGY_EDGES)
    delayed = openmc.DelayedGroupFilter(list(range(1, N_DELAYED + 1)))

    def tally_pair(name, filters, scores, estimator=None):
        pair = []
        for prefix, born_filters in (('born ', [born]), ('', [])):
            tally = openmc.Tally(name=prefix + name)
            tally.filters = born_filters + filters
            tally.scores = scores
            tally.estimator = estimator
            pair.append(tally)
        return pair

    model.tallies = (
        tally_pair('analog', [], ['absorption', 'nu-fission'], 'analog') +
        tally_pair('energy tracklength', [energy],
                   ['flux', 'absorption', 'nu-fission', 'prompt-nu-fission'],
                   'tracklength') +
        tally_pair('energy collision', [energy],
                   ['flux', 'absorption', 'nu-fission'], 'collision') +
        tally_pair('energy analog', [energy],
                   ['absorption', 'nu-fission', 'scatter'], 'analog') +
        tally_pair('delayed', [delayed], ['delayed-nu-fission'])
    )
    return model


def run_model(model, path):
    kwargs = {'cwd': path, 'openmc_exec': config['exe'],
              'event_based': config['event']}
    if config['mpi']:
        kwargs['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    return model.run(**kwargs)


def check_born_tallies(statepoint, expected_absorption):
    """Check the tallies of a run against the semantics of the filter.

    expected_absorption is the expected analog absorption per source neutron
    in each bin of the delayed group born filter.

    """
    def values(name, score, value='mean'):
        tally = statepoint.get_tally(name=name)
        return tally.get_values(scores=[score], value=value).ravel()

    # The filter is read back with its bins
    tally = statepoint.get_tally(name='born analog')
    born = tally.find_filter(openmc.DelayedGroupBornFilter)
    assert list(born.bins) == BORN_GROUPS
    df = tally.get_pandas_dataframe()
    assert list(df['delayedgroupborn'].unique()) == BORN_GROUPS

    # Analog absorption counts the neutrons born from each delayed group. The
    # bins add up to the tally without the filter.
    mean = values('born analog', 'absorption')
    std_dev = values('born analog', 'absorption', 'std_dev')
    total = values('analog', 'absorption')
    assert mean.sum() == pytest.approx(total[0], rel=1.0e-12)
    assert np.all(np.abs(mean - expected_absorption) <= 4.0 * std_dev), \
        f'{mean} differs from {expected_absorption} (std. dev. {std_dev})'
    assert np.all(std_dev <= 0.03 * expected_absorption)
    nu_fission = values('born analog', 'nu-fission')
    assert nu_fission.sum() == pytest.approx(
        values('analog', 'nu-fission')[0], rel=1.0e-12)

    # Every event in energy group j is caused by a neutron born from delayed
    # group j, for every estimator
    for estimator, scores in (
            ('tracklength', ['flux', 'absorption', 'nu-fission',
                             'prompt-nu-fission']),
            ('collision', ['flux', 'absorption', 'nu-fission']),
            ('analog', ['absorption', 'nu-fission', 'scatter'])):
        name = f'energy {estimator}'
        assert statepoint.get_tally(name='born ' + name).estimator == estimator
        for score in scores:
            by_born = values('born ' + name, score).reshape(
                len(BORN_GROUPS), N_GROUPS)
            by_energy = values(name, score)
            assert np.all(by_energy > 0.0)
            expected = np.zeros_like(by_born)
            expected[BORN_GROUPS, ENERGY_BIN] = by_energy[ENERGY_BIN]
            off_diagonal = expected == 0.0
            assert np.all(by_born[off_diagonal] == 0.0), \
                f'{name} {score}: events off the diagonal\n{by_born}'
            np.testing.assert_allclose(by_born, expected, rtol=1.0e-12)
            np.testing.assert_allclose(by_born.sum(axis=0), by_energy,
                                       rtol=1.0e-12)

    # The delayed production of each precursor group is beta_d times the
    # production of the neutrons of every birth group (track-length estimates
    # of both scores from the same tracks)
    by_born = values('born delayed', 'delayed-nu-fission').reshape(
        len(BORN_GROUPS), N_DELAYED)
    np.testing.assert_allclose(by_born.sum(axis=0),
                               values('delayed', 'delayed-nu-fission'),
                               rtol=1.0e-12)
    production = values('born energy tracklength', 'nu-fission').reshape(
        len(BORN_GROUPS), N_GROUPS).sum(axis=1)
    np.testing.assert_allclose(by_born, np.outer(production, BETA),
                               rtol=1.0e-12)


def test_delayed_group_born_eigenvalue(tmp_path):
    model = make_model(tmp_path / 'mgxs.h5')
    model.settings.run_mode = 'eigenvalue'
    model.settings.batches = 50
    model.settings.inactive = 5
    model.settings.particles = 10000

    with openmc.StatePoint(run_model(model, tmp_path)) as sp:
        k_collision = sp.global_tallies[0]
        assert k_collision['name'] == b'k-collision'
        assert abs(k_collision['mean'] - K_INF) <= 4.0 * k_collision['std_dev']

        # One absorption per source neutron, born prompt or from delayed group
        # d with probability 1 - beta and beta_d
        total = sp.get_tally(name='analog').get_values(scores=['absorption'])
        assert total.ravel()[0] == pytest.approx(1.0, rel=1.0e-12)
        check_born_tallies(sp, BIRTH_SPECTRUM)


def test_delayed_group_born_fixed_source(tmp_path):
    model = make_model(tmp_path / 'mgxs.h5')
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 50
    model.settings.particles = 20000

    # External source neutrons are born prompt; the fission neutrons are
    # transported from their sites with their delayed groups
    expected = FISSION_NEUTRONS * BIRTH_SPECTRUM
    expected[0] += 1.0
    with openmc.StatePoint(run_model(model, tmp_path)) as sp:
        check_born_tallies(sp, expected)
