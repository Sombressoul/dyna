import math
import torch
import torch.nn as nn

from typing import Optional, Union
from enum import Enum, auto as enum_auto

from dyna.lib.cpsf.functional.t_phc_fused import T_PHC_Fused
from dyna.lib.cpsf.functional.t_zero import T_Zero
from dyna.lib.cpsf.functional.sv_transform import spectrum_to_vector, vector_to_spectrum


class CPSFFusedCodebookMode(Enum):
    FAST = enum_auto()
    DUAL = enum_auto()


class CPSFCodebookFused(nn.Module):
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
        init_S_scale: float = 1.0e-3,
        phase_scale: float = 1.0,  # Use when T-value explodes.
        c_dtype: torch.dtype = torch.complex64,
        mode: CPSFFusedCodebookMode = CPSFFusedCodebookMode.FAST,
    ) -> None:
        super().__init__()

        if int(N) < 2:
            raise ValueError("N must be >= 2")
        if int(M) < 2:
            raise ValueError("M must be >= 2")
        if int(S) < 1:
            raise ValueError("S must be >= 1")
        if int(quad_nodes) < 2:
            raise ValueError("quad_nodes must be >= 2")
        if float(eps_total) <= 0.0:
            raise ValueError("eps_total must be > 0.")
        if float(overlap_rate) <= 0.0:
            raise ValueError("overlap_rate must be > 0.")
        if float(anisotropy) < 0.0:
            raise ValueError("anisotropy must be >= 0.")
        if float(init_S_scale) <= 0.0:
            raise ValueError("init_S_scale must be > 0.")
        if float(phase_scale) < 0.0:
            raise ValueError("phase_scale must be >= 0.")
        if c_dtype not in [torch.complex64, torch.complex128]:
            raise ValueError("c_dtype must be torch.complex64 or torch.complex128.")
        if not isinstance(mode, CPSFFusedCodebookMode):
            raise ValueError("mode must be CPSFFusedCodebookMode.")

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
        self.init_S_scale = float(init_S_scale)
        self.phase_scale = float(phase_scale)
        self.c_dtype = c_dtype
        self.r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64
        self.mode = mode

        self._init_sigmas()

        self.z_j = torch.nn.Parameter(
            data=(
                self._init_complex(
                    shape=[self.M, self.N],
                    unit=False,
                )
            )
            .mul(0.5)
            .detach(),
            requires_grad=True,
        )
        self.vec_d_j = torch.nn.Parameter(
            data=(
                self._init_complex(
                    shape=[self.M, self.N],
                    unit=True,
                )
            ).detach(),
            requires_grad=True,
        )
        self.T_hat_j = torch.nn.Parameter(
            data=self._init_complex(
                shape=[self.M, self.S],
                unit=False,
            )
            .mul(self.init_S_scale)
            .detach(),
            requires_grad=True,
        )
        self.alpha_j = torch.nn.Parameter(
            data=torch.empty([self.M], dtype=self.r_dtype)
            .uniform_(0.75, 1.25)
            .detach(),
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

        self.sigma_par = torch.nn.Parameter(
            data=torch.full((self.M,), sigma_par, dtype=self.r_dtype).detach(),
            requires_grad=True,
        )
        self.sigma_perp = torch.nn.Parameter(
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

        if unit:
            return self._to_unit(p)
        else:
            return p

    def _to_unit(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        n = torch.where(n.real == 0, torch.ones_like(n), n)
        x = (x / n).to(dtype=self.c_dtype)
        return x

    def _scale_phase(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        xr = x.real
        xi = x.imag * self.phase_scale
        return torch.complex(xr, xi).to(dtype=self.c_dtype)

    def _spectrum_to_vector(
        self,
        s: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.is_complex(s), "s should be complex"

        B, X = s.shape
        S = int(self.S)

        dtype_c = s.dtype
        dtype_r = torch.float64 if dtype_c == torch.complex128 else torch.float32
        device = s.device

        n = torch.arange(S, dtype=dtype_r, device=device)[:, None]
        k = torch.arange(X, dtype=dtype_r, device=device)[None, :]
        ang = (2.0 * math.pi / float(S)) * (n * k)
        A = torch.exp(1j * ang).to(dtype=dtype_c)
        x = (A @ s.transpose(0, 1)).transpose(0, 1)

        # project to real
        r = torch.abs(x)
        theta = torch.angle(x)
        x = r * theta.cos()

        return x

    def _vector_to_spectrum(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        assert not torch.is_complex(v), "v should be real"

        B, L = v.shape
        K = int(2 * self.N)

        dtype_c = (
            torch.complex128
            if v.dtype in (torch.float64, torch.double)
            else torch.complex64
        )
        dtype_r = torch.float64 if dtype_c == torch.complex128 else torch.float32
        device = v.device

        n = torch.arange(L, dtype=dtype_r, device=device)[:, None]
        k = torch.arange(K, dtype=dtype_r, device=device)[None, :]
        ang = (2.0 * math.pi / float(L)) * (n * k)
        A = torch.exp(1j * ang).to(dtype=dtype_c)

        b = v.to(dtype=dtype_c).transpose(0, 1)
        sol = torch.linalg.lstsq(A, b).solution
        x = sol.transpose(0, 1)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_complex():
            x = self._vector_to_spectrum(x)
        else:
            if x.shape[-1] != 2 * self.N:
                raise ValueError(
                    f"Complex input should have length 2N={2*self.N}, got: {x.shape[-1]}"
                )

        z = x[..., : self.N]
        vec_d = x[..., self.N :]

        z = self._scale_phase(z)
        vec_d = (
            self._to_unit(
                vec_d,
            )
            if self.autonorm_vec_d
            else vec_d
        )
        z_j = self._scale_phase(self.z_j)
        vec_d_j = (
            self._to_unit(
                self.vec_d_j,
            )
            if self.autonorm_vec_d_j
            else self.vec_d_j
        )
        T_hat_j = self.T_hat_j
        alpha_j = torch.clamp(
            self.alpha_j,
            min=self._get_tiny(self.alpha_j),
        )
        sigma_par = torch.clamp(
            self.sigma_par,
            min=self._get_tiny(self.sigma_par),
        )
        sigma_perp = torch.clamp(
            self.sigma_perp,
            min=self._get_tiny(self.sigma_perp),
        )

        if self.mode == CPSFFusedCodebookMode.FAST:
            B, N = z.shape
            M, S = T_hat_j.shape
            x = T_Zero(
                z=z,
                z_j=z_j.unsqueeze(0).expand([B, M, N]),
                vec_d=vec_d,
                vec_d_j=vec_d_j.unsqueeze(0).expand([B, M, N]),
                T_hat_j=T_hat_j.unsqueeze(0).expand([B, M, S]),
                alpha_j=alpha_j.unsqueeze(0).expand([B, M]),
                sigma_par=sigma_par.unsqueeze(0).expand([B, M]),
                sigma_perp=sigma_perp.unsqueeze(0).expand([B, M]),
            )
        elif self.mode == CPSFFusedCodebookMode.DUAL:
            x = T_PHC_Fused(
                z=z,
                vec_d=vec_d,
                z_j=z_j,
                vec_d_j=vec_d_j,
                T_hat_j=T_hat_j,
                alpha_j=alpha_j,
                sigma_par_j=sigma_par,
                sigma_perp_j=sigma_perp,
                quad_nodes=self.quad_nodes,
                eps_total=self.eps_total,
                n_chunk=self.n_chunk,
                m_chunk=self.m_chunk,
                dtype_override=self.c_dtype,
            )
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'")

        return self._spectrum_to_vector(x)
