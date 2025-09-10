import torch

from typing import Optional, Union

from dyna.lib.cpsf.structures import CPSFPsiOffsetsIterator
from dyna.lib.cpsf.functional.core_math import (
    # CPSF core math
    delta_vec_d,
    iota,
    lift,
    q,
    rho,
    R,
    R_ext,
    Sigma,
    # Field engine
    psi_over_offsets,
    T_classic_full,
    T_classic_window,
    Tau_nearest,
    Tau_dual,
    # Math helpers
    cholesky_spd,
    hermitianize,
)


class CPSFCore:
    def R(
        self,
        vec_d: torch.Tensor,
        kappa: float = 1.0e-3,
        sigma: float = 1.0e-3,
        jitter: float = 1.0e-6,
        p: float = 2.0,
        qmix: float = 2.0,
    ) -> torch.Tensor:
        return R(
            vec_d=vec_d,
            kappa=kappa,
            sigma=sigma,
            jitter=jitter,
            p=p,
            qmix=qmix,
        )

    def R_ext(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        return R_ext(
            R=R,
        )

    def Sigma(
        self,
        R_ext: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
    ) -> torch.Tensor:
        return Sigma(
            R_ext=R_ext,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
        )

    def q(
        self,
        w: torch.Tensor,
        R_ext: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
    ) -> torch.Tensor:
        return q(
            w=w,
            R_ext=R_ext,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
        )

    def delta_vec_d(
        self,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        return delta_vec_d(
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            eps=eps,
        )

    def iota(
        self,
        delta_z: torch.Tensor,
        delta_vec_d: torch.Tensor,
    ) -> torch.Tensor:
        return iota(
            delta_z=delta_z,
            delta_vec_d=delta_vec_d,
        )

    def lift(self, z: torch.Tensor) -> torch.Tensor:
        return lift(
            z=z,
        )

    def rho(
        self,
        q: torch.Tensor,
        q_max: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return rho(
            q=q,
            q_max=q_max,
        )

    def hermitianize(
        self,
        A: torch.Tensor,
    ) -> torch.Tensor:
        return hermitianize(
            A=A,
        )

    def cholesky_spd(
        self,
        A: torch.Tensor,
        eps: Optional[Union[float, torch.Tensor]] = None,
        use_jitter: bool = False,
    ) -> torch.Tensor:
        return cholesky_spd(
            A=A,
            eps=eps,
            use_jitter=use_jitter,
        )

    def Tau_nearest(
        self,
        z: torch.Tensor,
        z_j: torch.Tensor,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        T_hat_j: torch.Tensor,
        alpha_j: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
        R_j: Optional[torch.Tensor] = None,
        q_max: Optional[float] = None,
    ) -> torch.Tensor:
        return Tau_nearest(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            R_j=R_j,
            q_max=q_max,
        )

    def Tau_dual(
        self,
        z: torch.Tensor,
        z_j: torch.Tensor,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        T_hat_j: torch.Tensor,
        alpha_j: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
        k: torch.Tensor,
        R_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return Tau_dual(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            k=k,
            R_j=R_j,
        )

    def psi_over_offsets(
        self,
        z: torch.Tensor,
        z_j: torch.Tensor,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
        offsets: torch.Tensor,
        R_j: Optional[torch.Tensor] = None,
        q_max: Optional[float] = None,
    ) -> torch.Tensor:
        return psi_over_offsets(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets=offsets,
            R_j=R_j,
            q_max=q_max,
        )

    def T_classic_window(
        self,
        z: torch.Tensor,
        z_j: torch.Tensor,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        T_hat_j: torch.Tensor,
        alpha_j: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
        offsets_iterator: CPSFPsiOffsetsIterator,
        R_j: Optional[torch.Tensor] = None,
        q_max: Optional[float] = None,
    ) -> torch.Tensor:
        return T_classic_window(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets_iterator=offsets_iterator,
            R_j=R_j,
            q_max=q_max,
        )

    def T_classic_full(
        self,
        z: torch.Tensor,
        z_j: torch.Tensor,
        vec_d: torch.Tensor,
        vec_d_j: torch.Tensor,
        T_hat_j: torch.Tensor,
        alpha_j: torch.Tensor,
        sigma_par: torch.Tensor,
        sigma_perp: torch.Tensor,
        offsets_iterator: CPSFPsiOffsetsIterator,
        R_j: Optional[torch.Tensor] = None,
        q_max: Optional[float] = None,
        tol_abs: Optional[float] = None,
        tol_rel: Optional[float] = None,
        consecutive_below: int = 1,
    ) -> torch.Tensor:
        return T_classic_full(
            z=z,
            z_j=z_j,
            vec_d=vec_d,
            vec_d_j=vec_d_j,
            T_hat_j=T_hat_j,
            alpha_j=alpha_j,
            sigma_par=sigma_par,
            sigma_perp=sigma_perp,
            offsets_iterator=offsets_iterator,
            R_j=R_j,
            q_max=q_max,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            consecutive_below=consecutive_below,
        )
