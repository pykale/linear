import numpy as np
import pytest
from numpy import testing
from sklearn.exceptions import NotFittedError

from kalelinear.transformer import GFK
from kalelinear.transformer._gfk import _geodesic_flow_kernel, _pca_basis


@pytest.fixture(scope="module")
def gfk_data():
    rng = np.random.default_rng(0)
    n_source, n_target, n_features = 60, 80, 6

    X_source = rng.normal(size=(n_source, n_features)) * np.array([3.0, 2.0, 1.0, 0.3, 0.2, 0.1])
    X_target = rng.normal(size=(n_target, n_features)) * np.array([0.1, 0.3, 1.0, 2.0, 3.0, 0.2])
    domains = np.concatenate((np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)))
    X = np.vstack((X_source, X_target))
    return X, domains, X_source, X_target


def _numerical_kernel(source_basis, target_basis, n_steps=4000):
    """Integrate the geodesic flow numerically (twice the integral, as in Gong et al.)."""
    n_features = source_basis.shape[0]
    v1, cos_theta, vt = np.linalg.svd(source_basis.T @ target_basis, full_matrices=False)
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    u1 = source_basis @ v1
    target_rotated = target_basis @ vt.T
    residual = target_rotated - source_basis @ (source_basis.T @ target_rotated)
    t = (np.arange(n_steps) + 0.5) / n_steps
    kernel = np.zeros((n_features, n_features))
    for ti in t:
        phi = u1 * np.cos(ti * theta) + residual * (np.sin(ti * theta) / np.sin(theta))
        kernel += phi @ phi.T
    return 2 * kernel / n_steps


def test_gfk_kernel_matches_numerical_integration(gfk_data):
    _, _, X_source, X_target = gfk_data
    ps, pt = _pca_basis(X_source, 3), _pca_basis(X_target, 3)

    testing.assert_allclose(_geodesic_flow_kernel(ps, pt), _numerical_kernel(ps, pt), atol=1e-5)


def test_gfk_transform_reproduces_kernel(gfk_data):
    X, domains, _, _ = gfk_data
    gfk = GFK(n_components=3).fit(X, covariates=domains, target_covariate=1)
    z = gfk.transform(X)

    assert z.shape == X.shape
    testing.assert_allclose(z @ z.T, X @ gfk.kernel_ @ X.T, atol=1e-8)
    testing.assert_allclose(gfk.kernel_, gfk.kernel_.T)
    assert np.linalg.eigvalsh(gfk.kernel_).min() > -1e-10
    testing.assert_allclose(gfk.fit_transform(X, covariates=domains, target_covariate=1), z)


def test_gfk_identical_subspaces_gives_scaled_projection(gfk_data):
    _, _, X_source, _ = gfk_data
    ps = _pca_basis(X_source, 3)
    kernel = _geodesic_flow_kernel(ps, ps)

    assert np.isfinite(kernel).all()
    testing.assert_allclose(kernel, 2 * ps @ ps.T, atol=1e-10)


def test_gfk_near_zero_angles_are_continuous(gfk_data):
    _, _, X_source, _ = gfk_data
    ps = _pca_basis(X_source, 3)
    rng = np.random.default_rng(2)
    direction = rng.normal(size=ps.shape)
    kernels = []
    for eps in (1e-1, 2e-2, 1e-2, 5e-3, 1e-6):
        pt, _ = np.linalg.qr(ps + eps * direction)
        kernels.append(_geodesic_flow_kernel(ps, pt))
    for previous, current in zip(kernels, kernels[1:]):
        assert np.linalg.norm(current - previous) < 0.5
    testing.assert_allclose(kernels[-1], 2 * ps @ ps.T, atol=1e-4)


def test_gfk_attributes_and_feature_names(gfk_data):
    X, domains, _, _ = gfk_data
    gfk = GFK().fit(X, covariates=domains)

    assert gfk.n_components_ == X.shape[1] // 2
    assert gfk.principal_angles_.shape == (gfk.n_components_,)
    assert gfk.target_covariate_ == 1
    assert gfk.kernel_.shape == (X.shape[1], X.shape[1])
    testing.assert_array_equal(gfk.get_feature_names_out(), np.array([f"gfk{i}" for i in range(X.shape[1])]))


def test_gfk_transform_requires_fit(gfk_data):
    X, _, _, _ = gfk_data
    with pytest.raises(NotFittedError, match="not fitted"):
        GFK().transform(X)


def test_gfk_fit_requires_covariates(gfk_data):
    X, _, _, _ = gfk_data
    with pytest.raises(ValueError, match="Covariates must be provided"):
        GFK().fit(X)


def test_gfk_fit_requires_both_domains(gfk_data):
    X, _, _, _ = gfk_data
    with pytest.raises(ValueError, match="both source and target"):
        GFK().fit(X, covariates=np.zeros(X.shape[0], dtype=int))


def test_gfk_requires_two_samples_per_domain(gfk_data):
    X, _, _, _ = gfk_data
    with pytest.raises(ValueError, match="At least two target samples"):
        GFK().fit(X[:3], covariates=np.array([0, 0, 1]), target_covariate=1)


def test_gfk_validates_parameters(gfk_data):
    X, domains, _, _ = gfk_data
    with pytest.raises(ValueError, match="n_components"):
        GFK(n_components=0).fit(X, covariates=domains)
    with pytest.raises(ValueError, match="n_components"):
        GFK(n_components=X.shape[1] + 1).fit(X, covariates=domains)
    with pytest.raises(ValueError, match="target_covariate"):
        GFK().fit(X, covariates=domains, target_covariate=2)


def test_gfk_transform_validates_features(gfk_data):
    X, domains, _, _ = gfk_data
    gfk = GFK().fit(X, covariates=domains)
    with pytest.raises(ValueError, match="features"):
        gfk.transform(X[:, :2])
