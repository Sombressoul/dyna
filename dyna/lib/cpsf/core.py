import torch


class CPSFCore:
    def R(
        self,
        d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def R_ext(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def build_sigma(
        self,
        R_ext: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def cholesky_L(
        self,
        Sigma: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def delta_vec_d(
        self,
        d_q: torch.Tensor,
        d_j: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def lift(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def iota(
        self,
        dz: torch.Tensor,
        delta_d: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def tri_solve_norm_sq(
        self,
        L: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def rho_q(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
