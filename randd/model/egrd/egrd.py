import logging
import os
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from randd.model.base import GRD
from randd.model.linear import Linear


class EgrdCore:
    def __init__(
        self,
        r: NDArray, d: NDArray, d_measure: str, ndim: int,
        reps: NDArray, f0: NDArray, basis: NDArray
    ) -> None:
        # pre-process input
        self.reps, self.f0, self.H = self._validate_weights(reps, f0, basis)

        # obtain a continuous estimate of the bases
        self.f0_cont = Linear(self.reps, self.f0, d_measure=d_measure, ndim=ndim)
        self.basis_cont = [Linear(self.reps, self.H[:, i], d_measure=d_measure, ndim=ndim) for i in range(self.total_num_basis)]

        # Construct the difference matrix for the constraint in the quadratic programming
        c_opt = self._solve_qp(r, d)
        self.d_hat = self._reconstruct_egrd(c_opt)

    def _validate_weights(
        self, reps: NDArray, f0: NDArray, basis: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]:
        if reps.ndim == 0 or reps.ndim > 2:
            raise ValueError("Expected reps to be a 1d or 2d array, but got a {:d}d array instead.".format(reps.ndim))

        if reps.ndim == 1:
            reps = np.expand_dims(reps, axis=1)

        if f0.size != reps.shape[0]:
            raise ValueError("Expected {:d} entries in f0, but got {:d} entries instead.".format(reps.shape[0], f0.size))

        if basis.ndim != 2:
            raise ValueError("Expected a 2d array for basis, but got a {:d}d array".format(basis.ndim))

        if basis.shape[0] != reps.shape[0]:
            raise ValueError("Expected {:d} rows in basis, but got {:d} rows instead.".format(reps.shape[0], basis.shape[0]))

        self.total_num_basis = basis.shape[1]
        self.num_rep = reps.shape[0]

        self.num_resolution = 1
        if reps.shape[1] > 1:
            resolutions: NDArray = np.unique(reps[:, 1])
            self.num_resolution = resolutions.size

        self.num_bitrate = int(self.num_rep / self.num_resolution)

        if reps.shape[0] != self.num_resolution * self.num_bitrate:
            raise ValueError("Currently we do not support GRD basis sampled on an irregular grid.")

        f0 = f0.flatten()
        return reps, f0, basis

    def _solve_qp(self, r: NDArray, d: NDArray, num_basis: int = 10000) -> NDArray:
        num_samples = r.shape[0]
        num_basis = min(num_samples, self.total_num_basis, num_basis)
        if num_basis == 0:
            return np.array([0])

        # setup constraints
        self.D = self._construct_diff_mat()
        # setup objective function
        f0_tilde = self.f0_cont(r)
        f0f_tilde = f0_tilde - d
        H_tilde = np.zeros((num_samples, num_basis), dtype=float)

        for i in range(num_basis):
            H_tilde[:, i] = self.basis_cont[i](r)

        H = self.H[:, :num_basis]
        c_opt = self._solve_c(f0f_tilde, H_tilde, H)

        # error checking
        if np.any(np.isnan(c_opt)):
            logging.warning('Failed to solve the quadratic programming program. Return the default solution.')
            return np.array([0])

        return c_opt

    def _reconstruct_egrd(self, c: NDArray) -> NDArray:
        num_basis = c.shape[0]
        H = self.H[:, :num_basis]
        d_hat = np.dot(H, c) + self.f0
        return d_hat

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

    def sample(self) -> Tuple[NDArray, NDArray]:
        """Reconstruct a discrete GRD surface (or RD curve if only one resolution is provided)

        Args:
            reps (NDArray): (n,) or (n, 2) array. n is the number of sampled reps.
            q (NDArray): (n,) array.
            num_basis (int, optional): number of basis for GRD reconstruction. Defaults to None.

        Returns:
            FuncND: A GRD function continuous along the bitrate dimension
        """
        return self.reps, self.d_hat

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
        from osqp import OSQP, constant

        l_bound = -np.inf * np.ones(len(h))
        if A is not None:
            qp_A = np.vstack([G, A]).tocsc()
            qp_l = np.hstack([l_bound, b])
            qp_u = np.hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l_bound
            qp_u = h
        osqp_solver = OSQP()
        osqp_solver.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, eps_abs=1e-2, max_iter=10000000, verbose=False)
        if initvals is not None:
            osqp_solver.warm_start(x=initvals)
        res = osqp_solver.solve()
        if res.info.status_val != constant('OSQP_SOLVED'):
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


bases = {
    ('psnr', 1): 'psnr_100_6000_1080p.npz',
    ('psnr', 2): 'psnr_100_6000_allres.npz',
    ('ssimplus', 1): 'ssimplus_100_6000_1080p.npz',
    ('ssimplus', 2): 'ssimplus_100_6000_allres.npz',
    ('vmaf', 1): 'vmaf_100_6000_1080p.npz',
    ('vmaf', 2): 'vmaf_100_6000_allres.npz',
}


def egrd_factory(d_measure: str, ndim: int) -> EgrdCore:
    w_file = bases[(d_measure, ndim)]
    w_path = os.path.join(os.path.dirname(__file__), 'basis', w_file)
    weights = dict(np.load(w_path))
    return lambda r, d: EgrdCore(r=r, d=d, d_measure=d_measure, ndim=ndim, **weights)


class EGRD(GRD):
    r"""Eigen generalized rate-distortion function estimator.

    Args:
        r (NDArray): Encoding representations.
        d (NDArray): Corresponding distortions.
        d_measure (str): Name of the distortion measure.
        ndim (int): Number of dimensions of the RD function domain.

    References:
        Z. Duanmu, W. Liu, Z. Li, K. Ma, and Z. Wang,
        "Characterizing Generalized Rate-Distortion Performance of Video Coding: An Eigen Analysis Approach,"
        IEEE Transactions on Image Processing. vol. 29, pp. 6180-6193, 2020.
    """
    def __init__(self, r: NDArray, d: NDArray, d_measure: str, ndim: int) -> None:
        super().__init__(r, d, d_measure, ndim)
        # obtain dense samples on GRD
        egrd_cls = egrd_factory(d_measure=d_measure, ndim=ndim)
        model: EgrdCore = egrd_cls(r, d)
        _r, _d = model.sample()
        # get a continuous version
        dic = self._group_input(_r, _d)
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
