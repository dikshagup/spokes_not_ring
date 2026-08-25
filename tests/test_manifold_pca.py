"""PCA on toy arrays — exact known directions, round-trip, component cap."""

import numpy as np

from weekday_manifold.manifold.pca import PCA, cap_components, fit_pca, group_centroids


def test_pca_recovers_dominant_axis():
    # Points spread mostly along x, a little along y.
    rng = np.random.default_rng(0)
    X = np.zeros((200, 3))
    X[:, 0] = rng.normal(0, 10, 200)
    X[:, 1] = rng.normal(0, 1, 200)
    pca = fit_pca(X, n_components=3)
    top = np.abs(pca.components[0])
    assert top[0] > 0.95  # first PC ~ x-axis
    # Variance ratio is dominated by PC1.
    assert pca.explained_variance_ratio[0] > 0.9
    assert np.isclose(pca.explained_variance_ratio.sum(), 1.0, atol=1e-6) or \
        pca.explained_variance_ratio.sum() <= 1.0 + 1e-6


def test_pca_round_trip_in_full_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    pca = fit_pca(X, n_components=4)
    Z = pca.transform(X)
    Xr = pca.inverse_transform(Z)
    assert np.allclose(X, Xr, atol=1e-8)


def test_cap_components():
    # min(n_components, n_samples-1, n_features)
    assert cap_components(64, 10, 1600) == 9
    assert cap_components(64, 100, 1600) == 64
    assert cap_components(64, 100, 32) == 32
    assert cap_components(64, 1, 1600) == 1  # floor at 1


def test_cap_components_none_is_full_valid_rank():
    # None -> as many as valid = min(n_samples-1, n_features)
    assert cap_components(None, 14, 1600) == 13
    assert cap_components(None, 100, 32) == 32
    assert cap_components(None, 1, 1600) == 1


def test_fit_pca_none_keeps_full_rank():
    X = np.random.default_rng(3).normal(size=(14, 1600))
    pca = fit_pca(X, n_components=None)
    assert pca.n_components == 13  # n_samples - 1


def test_fit_pca_caps_components_for_few_samples():
    X = np.random.default_rng(2).normal(size=(8, 1600))
    pca = fit_pca(X, n_components=64)
    assert pca.n_components == 7  # n_samples - 1


def test_group_centroids_means_per_label():
    Z = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 8.0]])
    labels = np.array([0, 0, 1, 1])
    c = group_centroids(Z, labels, n_groups=2)
    assert np.allclose(c[0], [1.0, 0.0])
    assert np.allclose(c[1], [0.0, 6.0])


def test_group_centroids_empty_group_raises():
    Z = np.zeros((2, 2))
    labels = np.array([0, 0])
    try:
        group_centroids(Z, labels, n_groups=3)
        assert False, "expected ValueError for empty group"
    except ValueError:
        pass
