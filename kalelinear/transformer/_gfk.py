# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Geodesic Flow Kernel (GFK)."""

from numbers import Integral

import numpy as np
from sklearn.base import _fit_context, BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from kalelinear._domain import check_binary_domain_covariates, split_domain_indices
from kalelinear.transformer._coral import _symmetric_sqrt

# Below this principal angle (radians) the closed-form weights are replaced by
# their Taylor expansions to avoid division by (near-)zero angles.
_SMALL_ANGLE = 1e-2


def _pca_basis(X, n_components):
    """Return the top ``n_components`` principal directions of ``X`` as columns."""
    _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    return vt[:n_components].T


def _geodesic_flow_kernel(source_basis, target_basis):
    """Return the geodesic flow kernel matrix between two subspaces.

    Both inputs are orthonormal bases of shape ``(n_features, n_components)``.
    The kernel is computed in closed form from the principal angles between
    the subspaces, without forming the orthogonal complement of the source
    subspace explicitly. Principal angles close to zero are handled with
    Taylor expansions, so identical or overlapping subspaces are supported.
    """
    # Principal angles: Ps^T Pt = V1 cos(Theta) V^T
    v1, cos_theta, vt = np.linalg.svd(source_basis.T @ target_basis, full_matrices=False)
    v = vt.T
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Component of the target basis orthogonal to the source subspace.
    u1 = source_basis @ v1
    target_rotated = target_basis @ v
    residual = target_rotated - source_basis @ (source_basis.T @ target_rotated)

    small = theta < _SMALL_ANGLE
    safe_theta = np.where(small, 1.0, theta)

    # sin(2θ) / (2θ)
    sinc_2theta = np.sinc(2 * theta / np.pi)
    lambda1 = 1.0 + sinc_2theta
    # sin(θ) / θ
    ratio = np.sinc(theta / np.pi)
    # (2θ - sin(2θ)) / (2 sin²θ θ) -> 2/3 as θ -> 0
    w_direct = (2 * safe_theta - np.sin(2 * safe_theta)) / (2 * safe_theta * np.sin(safe_theta) ** 2)
    w_series = (2 / 3 - 2 * theta**2 / 15 + 4 * theta**4 / 315) / (1 - theta**2 / 3 + 2 * theta**4 / 45)
    weight = np.where(small, w_series, w_direct)

    cross = residual * ratio
    kernel = (u1 * lambda1) @ u1.T + u1 @ cross.T + cross @ u1.T + (residual * weight) @ residual.T
    return (kernel + kernel.T) / 2


class GFK(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Geodesic Flow Kernel (GFK) for unsupervised domain adaptation.

    GFK models the source and target domains as points (subspaces spanned by
    their leading principal components) on a Grassmann manifold and
    integrates over all intermediate subspaces along the geodesic flow
    connecting them. The resulting positive semi-definite kernel has a closed
    form in terms of the principal angles :math:`\\theta_i` between the two
    subspaces:

    .. math::

        G = Q \\begin{bmatrix} V_1 & 0 \\\\ 0 & V_2 \\end{bmatrix}
        \\begin{bmatrix} \\Lambda_1 & \\Lambda_2 \\\\ \\Lambda_2 & \\Lambda_3 \\end{bmatrix}
        \\begin{bmatrix} V_1 & 0 \\\\ 0 & V_2 \\end{bmatrix}^\\top Q^\\top,

    with :math:`\\lambda_{1i} = 1 + \\sin(2\\theta_i) / (2\\theta_i)`,
    :math:`\\lambda_{2i} = (\\cos(2\\theta_i) - 1) / (2\\theta_i)` and
    :math:`\\lambda_{3i} = 1 - \\sin(2\\theta_i) / (2\\theta_i)`. The kernel
    is evaluated stably for principal angles close to zero.

    :meth:`transform` embeds samples as :math:`G^{1/2} x`, so that inner
    products in the embedded space equal the geodesic flow kernel. The same
    mapping is applied to source and target samples.

    ``covariates`` represent binary domain labels of length ``n_samples``.
    They must contain both source and target domains during :meth:`fit`.
    ``target_covariate`` selects which label is treated as the target domain.

    Parameters
    ----------
    n_components : int, default=None
        Dimensionality of the source and target subspaces. It must not exceed
        ``min(n_features, n_source - 1, n_target - 1)``. If ``None``, it is
        set to ``max(1, n_features // 2)`` capped by the above limit.

    Attributes
    ----------
    source_basis_ : ndarray of shape (n_features, n_components_)
        Orthonormal basis of the source PCA subspace.
    target_basis_ : ndarray of shape (n_features, n_components_)
        Orthonormal basis of the target PCA subspace.
    principal_angles_ : ndarray of shape (n_components_,)
        Principal angles (radians) between the source and target subspaces.
    kernel_ : ndarray of shape (n_features, n_features)
        Geodesic flow kernel matrix :math:`G`.
    transform_matrix_ : ndarray of shape (n_features, n_features)
        Symmetric square root :math:`G^{1/2}` used by :meth:`transform`.
    n_components_ : int
        Effective subspace dimensionality.
    target_covariate_ : object
        Domain label treated as the target domain.

    References
    ----------
    Gong, B., Shi, Y., Sha, F. and Grauman, K., 2012. Geodesic flow kernel
    for unsupervised domain adaptation. In *IEEE Conference on Computer
    Vision and Pattern Recognition* (pp. 2066-2073).
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(self, n_components=None):
        self.n_components = n_components

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, covariates=None, target_covariate=None):
        """Fit GFK on source and target samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Source and target samples stacked along the first axis.
        y : array-like, default=None
            Ignored. Present for scikit-learn API consistency. GFK is an
            unsupervised domain adaptation method.
        covariates : array-like of shape (n_samples,), default=None
            Binary domain label for every sample. Must contain both source
            and target domain values.
        target_covariate : object, default=None
            Domain label treated as the target domain. Defaults to the last
            unique covariate value, as in the other KaleLinear adapters.

        Returns
        -------
        self : GFK
            Fitted transformer.
        """
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape

        if covariates is None:
            raise ValueError("Covariates must be provided for GFK during `fit`.")

        covariates, unique_covariates = check_binary_domain_covariates(
            covariates,
            n_samples,
            require_numeric=True,
            error_prefix=f"Covariates for {self.__class__.__name__}",
            both_domains_message=(
                f"Covariates for {self.__class__.__name__} must contain both source and target domain values."
            ),
        )
        split = split_domain_indices(covariates, target_covariate)

        X_source = X[split.source_idx]
        X_target = X[split.target_idx]
        if X_source.shape[0] < 2:
            raise ValueError("At least two source samples are required to estimate the source subspace.")
        if X_target.shape[0] < 2:
            raise ValueError("At least two target samples are required to estimate the target subspace.")

        max_components = min(n_features, X_source.shape[0] - 1, X_target.shape[0] - 1)
        if self.n_components is None:
            n_components = min(max(1, n_features // 2), max_components)
        elif self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} must be less than or equal to "
                f"min(n_features, n_source - 1, n_target - 1) = {max_components}."
            )
        else:
            n_components = self.n_components

        self.source_basis_ = _pca_basis(X_source, n_components)
        self.target_basis_ = _pca_basis(X_target, n_components)
        cos_theta = np.linalg.svd(self.source_basis_.T @ self.target_basis_, compute_uv=False)
        self.principal_angles_ = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        self.kernel_ = _geodesic_flow_kernel(self.source_basis_, self.target_basis_)
        self.transform_matrix_ = _symmetric_sqrt(self.kernel_)
        self.n_components_ = n_components
        self.target_covariate_ = split.target_covariate
        self.domain_values_ = unique_covariates
        self._n_features_out = n_features
        return self

    def transform(self, X):
        """Embed samples with the square root of the geodesic flow kernel.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform. The same mapping applies to source and
            target samples.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed samples. Feature dimensionality is preserved.
        """
        check_is_fitted(self, "kernel_")
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        return X @ self.transform_matrix_

    def fit_transform(self, X, y=None, covariates=None, target_covariate=None):
        """Fit GFK and transform the source and target samples.

        Parameters are the same as for :meth:`fit`.
        """
        return self.fit(X, y=y, covariates=covariates, target_covariate=target_covariate).transform(X)
