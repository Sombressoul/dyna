import math
import torch
import torch.nn as nn

from typing import Optional, Union

from dyna.lib.cpsf.functional.t_phc_fused import T_PHC_Fused


class CPSFFusedCodebook(nn.Module):
    def __init__(
        self,
        N: int,
        M: int,
        S: int,
        quad_nodes: int = 8,
        n_chunk: Optional[int] = None,
        m_chunk: Optional[int] = None,
        eps_total: float = 1.0e-3,
        autonorm_vec_d: bool = True,
        autonorm_vec_d_j: bool = True,
        overlap_rate: float = 0.25,
        anisotropy: float = 0.75,
        c_dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()

        if float(overlap_rate) <= 0.0:
            raise ValueError("overlap_rate must be > 0.")
        if float(anisotropy) < 0.0:
            raise ValueError("anisotropy must be >= 0.")

        self.N = int(N)
        self.M = int(M)
        self.S = int(S)
        self.quad_nodes = int(quad_nodes)
        self.n_chunk = int(n_chunk) if n_chunk is not None else self.N
        self.m_chunk = int(m_chunk) if m_chunk is not None else self.M
        self.eps_total = float(eps_total)
        self.autonorm_vec_d = bool(autonorm_vec_d)
        self.autonorm_vec_d_j = bool(autonorm_vec_d_j)
        self.overlap_rate = float(overlap_rate)
        self.anisotropy = float(anisotropy)
        self.c_dtype = c_dtype
        self.r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64

        self._init_sigmas()

        self.z_j = torch.nn.Parameter(
            data=(
                self._init_complex(
                    shape=[self.M, self.N],
                    unit=False,
                )
                / 2.0
            ).detach(),
            requires_grad=True,
        )
        self.vec_d_j = torch.nn.Parameter(
            data=(
                self._init_complex(
                    shape=[self.M, self.N],
                    unit=True,
                )
                / 2.0
            ).detach(),
            requires_grad=True,
        )
        self.T_hat_j = torch.nn.Parameter(
            data=self._init_complex(
                shape=[self.M, self.S],
                unit=False,
            ).detach(),
            requires_grad=True,
        )
        self.alpha_j = torch.nn.Parameter(
            data=torch.empty([self.M], dtype=self.r_dtype).uniform_(0.5, 1.5).detach(),
            requires_grad=True,
        )

    def _init_sigmas(
        self,
    ) -> None:
        N = float(self.N)
        M = float(max(self.M, 1))
        rho = 1.0 + self.anisotropy
        V_N = (math.pi ** (N * 0.5)) / math.gamma(N * 0.5 + 1.0)

        r_perp = (self.overlap_rate / (M * V_N * rho)) ** (1.0 / N)
        if not (r_perp > 0.0):
            raise ValueError("Computed r_perp is non-positive; check overlap_rate/M/N.")

        sigma_perp = 2.0 * math.pi * (r_perp**2)
        sigma_par = (rho**2) * sigma_perp

        self.sigma_par_j = torch.nn.Parameter(
            data=torch.full((self.M,), sigma_par, dtype=self.r_dtype).detach(),
            requires_grad=True,
        )
        self.sigma_perp_j = torch.nn.Parameter(
            data=torch.full((self.M,), sigma_perp, dtype=self.r_dtype).detach(),
            requires_grad=True,
        )

    def _get_tiny(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.as_tensor(
            torch.finfo(x.dtype).tiny, device=x.device, dtype=x.dtype
        )

    def _init_complex(
        self,
        shape: Union[torch.Size, list[int]],
        unit: bool = False,
    ) -> torch.Tensor:
        xr = torch.empty(shape, dtype=self.r_dtype).uniform_(-1.0, +1.0)
        xi = torch.empty(shape, dtype=self.r_dtype).uniform_(-1.0, +1.0)
        p = torch.complex(
            real=xr,
            imag=xi,
        ).to(dtype=self.c_dtype)

        if not unit:
            return p

        n = torch.linalg.vector_norm(p, dim=-1, keepdim=True)
        n = torch.where(n.real == 0, torch.ones_like(n), n)
        p = (p / n).to(dtype=self.c_dtype)

        return p

    def _to_unit(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        n = torch.where(n.real == 0, torch.ones_like(n), n)
        x = (x / n).to(dtype=self.c_dtype)
        return x

    def forward(
        self,
        z: torch.Tensor,
        vec_d: torch.Tensor,
    ) -> torch.Tensor:
        if not z.is_complex():
            raise ValueError(f"'z' should be complex, got: '{type(z)}'")
        if not vec_d.is_complex():
            raise ValueError(f"'vec_d' should be complex, got: '{type(vec_d)}'")

        return T_PHC_Fused(
            z=z,
            vec_d=(
                self._to_unit(
                    vec_d,
                )
                if self.autonorm_vec_d
                else vec_d
            ),
            z_j=self.z_j,
            vec_d_j=(
                self._to_unit(
                    self.vec_d_j,
                )
                if self.autonorm_vec_d_j
                else self.vec_d_j
            ),
            T_hat_j=self.T_hat_j,
            alpha_j=torch.clamp(
                self.alpha_j,
                min=self._get_tiny(self.alpha_j),
            ),
            sigma_par_j=torch.clamp(
                self.sigma_par_j,
                min=self._get_tiny(self.sigma_par_j),
            ),
            sigma_perp_j=torch.clamp(
                self.sigma_perp_j,
                min=self._get_tiny(self.sigma_perp_j),
            ),
            quad_nodes=self.quad_nodes,
            eps_total=self.eps_total,
            n_chunk=self.n_chunk,
            m_chunk=self.m_chunk,
            dtype_override=self.c_dtype,
        )
