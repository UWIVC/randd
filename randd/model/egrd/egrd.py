import numpy as np
from pathlib import Path
from randd.model.base import GRD
from scipy.interpolate import interp1d
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Union, Optional, Dict, Type


def egrd_factory(ndim: int, d_measure: str = 'ssim') -> EgrdCore:
    pass


class EGRD(GRD):
    def __init__(self, r: NDArray, d: NDArray) -> None:
        # obtain dense samples on GRD
        egrd = egrd_factory()
        _r, _d = egrd(r, d)
        dic = self._group_input(r, d)
        self.f = {}
        for key in dic:
            ri, di = dic[key]
            self.f[key] = interp1d(ri, di, fill_value='extrapolate')

    def __call__(self, r: NDArray) -> NDArray:
        d = np.zeros(r.shape[0])
        for i, row in enumerate(r):
            row = np.expand_dims(row, axis=0)
            dic: Dict = self._group_input(row)
            hparam, value = dic.popitem()
            rate, _ = value
            d[i] = self.f[hparam](rate[0]) if hparam in self.f else np.nan

        return d


# class Temp:
#     def __init__(
#         self,
#         reps: NDArray,
#         f0: NDArray,
#         basis: NDArray,
#         base_estimator1d: BaseEstimator1d = LinearEstimator1d()
#     ) -> None:
#         self.egrd_core = EgrdCore(reps, f0, basis)
#         self.base_estimator1d = base_estimator1d

#     def __call__(self, reps: NDArray, q: NDArray, num_basis: Union[int, None] = None) -> Tuple[Func1d, Func1d]:
#         q_hat = self.egrd_core.recon_discrete_grd(reps, q, num_basis)
#         if self.egrd_core.num_resolution > 1:
#             grd2d = GRDSurface2d(self.egrd_core.reps, q_hat, self.base_estimator1d.estimate_rq_func)
#             q_hat, r_hat, _ = grd2d.get_rq_envelop()
#         else:
#             r_hat = self.egrd_core.reps

#         r_hat: NDArray = r_hat.flatten()
#         q_hat: NDArray = q_hat.flatten()

#         rq_func, qr_func = self.base_estimator1d(r_hat, q_hat)

#         return rq_func, qr_func


class EgrdCore:
    def __init__(self, reps: NDArray, f0: NDArray, basis: NDArray) -> None:
        """Perform EGRD estimation with the given average function and basis

        Args:
            reps (NDArray): (N,) or (N, 1) or (N, 2) array.
                N is the number of representations.
                First dimension is bitrate, and second dimension is resolution.
            f0 (NDArray): (N,) or (N, 1) array.
            basis (NDArray): (N, M) array. M is the total number of basis.
        """

        # Check compatibility of shapes of arguments
        if reps.ndim == 0 or reps.ndim > 2:
            raise ValueError("Expected reps to be a 1d or 2d array, but got a {:d}d array instead.".format(reps.ndim))
        self.num_rep = reps.shape[0]
        if reps.ndim == 2 and reps.shape[1] == 1:
            self.reps = reps.flatten()
        else:
            self.reps = reps

        if f0.size != self.num_rep:
            raise ValueError("Expected {:d} entries in f0, but got {:d} entries instead.".format(self.num_rep, f0.size))
        self.f0 = f0.flatten()

        if basis.ndim != 2:
            raise ValueError("Expected a 2d array for basis, but got a {:d}d array".format(basis.ndim))
        if basis.shape[0] != self.num_rep:
            raise ValueError("Expected {:d} rows in basis, but got {:d} rows instead.".format(self.num_rep, basis.shape[0]))
        self.H = basis
        self.total_num_basis = self.H.shape[1]

        # Determine number of resolutions
        if self.reps.ndim == 1:
            self.num_resolution = 1
        elif self.reps.shape[1] == 2:
            resolutions: NDArray = np.unique(self.reps[:, 1])
            self.num_resolution = resolutions.size
        else:
            raise ValueError("Rep specification other than bitrate and resolution is not supported")
        # Determine number of bitrates
        self.num_bitrate = int(self.num_rep / self.num_resolution)

        # Check if sampled on a regular grid when 2d GRD
        assert self.num_rep == self.num_resolution * self.num_bitrate, "Currently we do not support GRD basis sampled on an irregular grid."

        if self.reps.ndim == 1:
            self.continuous_f0 = Func1d(scipy.interpolate.interp1d(
                self.reps.flatten(), self.f0.flatten(), fill_value='extrapolate'))
            self.continuous_basis = [Func1d(scipy.interpolate.interp1d(
                self.reps.flatten(), self.H[:, i], fill_value='extrapolate')) for i in range(self.total_num_basis)]
        else:
            self.continuous_f0 = Func2d(GRDSurface2d(self.reps, self.f0))
            self.continuous_basis = [Func2d(GRDSurface2d(self.reps, self.H[:, i])) for i in range(self.total_num_basis)]

        # Construct the difference matrix for the constraint in the quadratic programming
        self.D = self._construct_diff_mat()

    def _construct_diff_mat(self):
        '''Construct discrete derivative matrices D in Eq.X'''
        # Construct D1
        D1_t1 = np.diag(np.ones(self.num_rep))
        D1_t2 = np.ones(self.num_rep - 1) * - 1  # Left elements of diagonal are -1.
        D1_t3 = np.diag(D1_t2, k=-1)  # Diagonal matrix shifted to left.
        D1_t4 = D1_t1 + D1_t3
        idx_rmv = np.array(np.arange(0, self.num_rep))[np.s_[::self.num_bitrate]]
        D = np.delete(D1_t4, idx_rmv, 0)

        # Construct D2
        if self.num_resolution > 1:
            D2 = np.zeros((self.num_resolution - 1, self.num_rep))
            for idx, row in enumerate(D2):
                row[(idx + 1) * self.num_bitrate - 1] = -1
                row[(idx + 2) * self.num_bitrate - 1] = 1

            D = np.concatenate((D, D2), axis=0)

        return D

    def recon_discrete_grd(self, reps: NDArray, q: NDArray, num_basis: Optional[int] = None) -> NDArray:
        """Reconstruct a discrete GRD surface (or RD curve if only one resolution is provided)

        Args:
            reps (NDArray): (n,) or (n, 2) array. n is the number of sampled reps.
            q (NDArray): (n,) array.
            num_basis (int, optional): number of basis for GRD reconstruction. Defaults to None.

        Returns:
            NDArray: (self.num_rep, ) array. The values of the reconstructed GRD function at self.rep.
        """
        if (self.reps.ndim == 1 and reps.ndim != 1):
            raise ValueError("Expected reps to be a 1-d array, but got a {:d}-d array instead.".format(reps.ndim))

        if (self.reps.ndim == 2 and (reps.ndim != 2 or reps.shape[1] != 2)):
            raise ValueError("Expected reps to be an n-by-2 array, but got a {} array instead".format(reps.shape))

        num_samples = reps.shape[0]
        if num_basis is None:
            num_basis = min(num_samples, self.total_num_basis)
        else:
            num_basis = min(num_samples, self.total_num_basis, num_basis)

        f0_tilde = self.continuous_f0(reps)
        f0f_tilde = f0_tilde - q

        if num_basis > 0:
            H_tilde = np.zeros((num_samples, num_basis), dtype=np.float)
            for i in range(num_basis):
                H_tilde[:, i] = self.continuous_basis[i](reps)

            H = self.H[:, :num_basis]
            c_opt = self._solve_c(f0f_tilde, H_tilde, H)
            if np.any(np.isnan(c_opt)):
                raise ValueError('The rate-quality curve cannot be estimated!')
            else:
                q_hat = np.dot(H, c_opt) + self.f0
        else:
            q_hat = self.f0

        return q_hat

    def __call__(self, reps: NDArray, q: NDArray, num_basis: Optional[int] = None) -> FuncNd:
        """Reconstruct a discrete GRD surface (or RD curve if only one resolution is provided)

        Args:
            reps (NDArray): (n,) or (n, 2) array. n is the number of sampled reps.
            q (NDArray): (n,) array.
            num_basis (int, optional): number of basis for GRD reconstruction. Defaults to None.

        Returns:
            FuncND: A GRD function continuous along the bitrate dimension
        """
        q_hat = self.recon_discrete_grd(reps, q, num_basis)
        if self.reps.ndim == 1:
            grd_func = scipy.interpolate.interp1d(self.reps, q_hat, fill_value='extrapolate')
        else:
            grd_func = Func2d(GRDSurface2d(self.reps, q_hat))

        return grd_func

    def _osqp_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
            minimize
                (1/2) * x.T * P * x + q.T * x
            subject to
                G * x <= h
                A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        Parameters
        ----------
        P : scipy.sparse.csc_matrix
            Symmetric quadratic-cost matrix.
        q : numpy.array
            Quadratic cost vector.
        G : scipy.sparse.csc_matrix
            Linear inequality constraint matrix.
        h : numpy.array
            Linear inequality constraint vector.
        A : scipy.sparse.csc_matrix, optional
            Linear equality constraint matrix.
        b : numpy.array, optional
            Linear equality constraint vector.
        initvals : numpy.array, optional
            Warm-start guess vector.
        Returns
        -------
        x : array, shape=(n,)
            Solution to the QP, if found, otherwise ``None``.
        Note
        ----
        OSQP requires `P` to be symmetric, and won't check for errors otherwise.
        Check out for this point if you e.g. `get nan values
        <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
        """
        from osqp import OSQP

        l_bound = -np.inf * np.ones(len(h))
        if A is not None:
            qp_A = np.vstack([G, A]).tocsc()
            qp_l = np.hstack([l_bound, b])
            qp_u = np.hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l_bound
            qp_u = h
        osqp = OSQP()
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, eps_abs=1e-2, max_iter=10000000, verbose=False)
        if initvals is not None:
            osqp.warm_start(x=initvals)
        res = osqp.solve()
        if res.info.status_val != osqp.constant('OSQP_SOLVED'):
            print("OSQP exited with status '%s'" % res.info.status)
        return res.x

    def _solve_c(self, f0f_tilde, H_tilde, H):
        r'''Construct input for quadratic programming and solve for c'''
        from scipy.sparse import csc_matrix

        P = 2 * np.dot(H_tilde.T, H_tilde)
        q = np.dot(2 * f0f_tilde, H_tilde)
        G = np.dot(-self.D, H)
        h = np.dot(self.D, self.f0)

        P = csc_matrix(P)
        G = csc_matrix(G)
        c_opt = self._osqp_solve_qp(P, q, G=G, h=h)
        return c_opt


def generate_eigen_basis(data: ArrayLike, return_eigval: bool = False) -> Tuple[NDArray, ...]:
    data = np.array(data, dtype=np.float_)
    assert data.ndim == 2, "Expected data to be a 2D array, but got a {:d}d array instead."
    f0 = data.mean(axis=0)
    data = data - f0
    C = np.cov(data.T)
    eig_values, H = np.linalg.eigh(C)
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    H = H[:, idx]
    if return_eigval:
        return f0, H, eig_values
    else:
        return f0, H
