import numpy as np

from odtoy import grid_network
from odtoy.days import gravity_od, make_prior, sample_day_od


def test_gravity_od_shape_and_total():
    net = grid_network(4, 5)
    od = gravity_od(net, cols=5, total_trips=1000.0)
    assert od.shape == (20, 20)
    assert np.isclose(od.sum(), 1000.0)
    assert np.all(od >= 0)
    assert np.all(np.diag(od) == 0)


def test_gravity_od_prefers_near_zones():
    net = grid_network(4, 4, cap_spread=0.0)
    od = gravity_od(net, cols=4, total_trips=1000.0, decay=1.0, seed=0)
    # zone 0 is the corner: its neighbour (zone 1) should attract more
    # than the opposite corner (zone 15), whatever the random attractions
    assert od[0, 1] > od[0, 15]


def test_gravity_od_reproducible():
    net = grid_network(3, 3)
    a = gravity_od(net, cols=3, total_trips=100.0, seed=5)
    b = gravity_od(net, cols=3, total_trips=100.0, seed=5)
    assert np.array_equal(a, b)


def test_sample_day_od_is_centred_on_mean():
    net = grid_network(4, 4)
    od_mean = gravity_od(net, cols=4, total_trips=2000.0)
    rng = np.random.default_rng(0)
    days = np.stack([sample_day_od(od_mean, rng) for _ in range(300)])
    ratio = days.mean(axis=0)[od_mean > 0] / od_mean[od_mean > 0]
    assert np.all(np.diag(days.mean(axis=0)) == 0)
    assert np.all(days >= 0)
    # log-normal with small sd: mean ratio close to 1 (slightly above)
    assert 0.9 < np.median(ratio) < 1.15


def test_sample_day_od_zero_sd_returns_mean():
    net = grid_network(3, 3)
    od_mean = gravity_od(net, cols=3, total_trips=100.0)
    rng = np.random.default_rng(0)
    od = sample_day_od(od_mean, rng, scale_sd=0.0, zone_sd=0.0, cell_sd=0.0)
    assert np.allclose(od, od_mean)


def test_sample_day_od_days_differ():
    net = grid_network(3, 3)
    od_mean = gravity_od(net, cols=3, total_trips=100.0)
    rng = np.random.default_rng(1)
    a, b = sample_day_od(od_mean, rng), sample_day_od(od_mean, rng)
    assert not np.allclose(a, b)


def test_make_prior_is_biased_but_correlated():
    net = grid_network(4, 4)
    od_mean = gravity_od(net, cols=4, total_trips=2000.0)
    prior = make_prior(od_mean, np.random.default_rng(0), scale_bias=0.8)
    mask = od_mean > 0
    assert np.all(prior >= 0)
    assert np.all(np.diag(prior) == 0)
    # global bias shows up in the total
    assert 0.6 < prior.sum() / od_mean.sum() < 1.0
    # but the spatial structure is preserved
    corr = np.corrcoef(np.log(prior[mask]), np.log(od_mean[mask]))[0, 1]
    assert corr > 0.8


def test_make_prior_no_error_returns_scaled_mean():
    net = grid_network(3, 3)
    od_mean = gravity_od(net, cols=3, total_trips=100.0)
    prior = make_prior(od_mean, np.random.default_rng(0), scale_bias=0.5, zone_sd=0.0, cell_sd=0.0)
    assert np.allclose(prior, 0.5 * od_mean)
