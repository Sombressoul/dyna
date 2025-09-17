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
        c_dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()

        self.N = int(N)
        self.M = int(M)
        self.S = int(S)
        self.quad_nodes = int(quad_nodes)
        self.n_chunk = int(n_chunk) if n_chunk is not None else self.N
        self.m_chunk = int(m_chunk) if m_chunk is not None else self.M
        self.eps_total = float(eps_total)
        self.c_dtype = c_dtype
        self.r_dtype = torch.float32 if c_dtype == torch.complex64 else torch.float64

        self.z_j = torch.nn.Parameter(
            data=self._init_complex(
                shape=[self.M, self.N],
                unit=False,
            ),
            requires_grad=True,
        )
        self.vec_d_j = torch.nn.Parameter(
            data=self._init_complex(
                shape=[self.M, self.N],
                unit=True,
            ),
            requires_grad=True,
        )
        self.T_hat_j = torch.nn.Parameter(
            data=self._init_complex(
                shape=[self.M, self.S],
                unit=False,
            ),
            requires_grad=True,
        )
        self.alpha_j = torch.nn.Parameter(
            data=torch.rand([self.M], dtype=self.r_dtype),
            requires_grad=True,
        )
        self.sigma_par_j = torch.nn.Parameter(
            data=torch.empty([self.M], dtype=self.r_dtype).uniform_(0.5, 1.5),
            requires_grad=True,
        )
        self.sigma_perp_j = torch.nn.Parameter(
            data=torch.empty([self.M], dtype=self.r_dtype).uniform_(0.5, 1.5),
            requires_grad=True,
        )

    def _init_complex(
        self,
        shape: Union[torch.Size, list[int]],
        unit: bool = False,
    ) -> torch.Tensor:
        xr = torch.randn(shape, dtype=self.r_dtype)
        xi = torch.randn(shape, dtype=self.r_dtype)
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
            vec_d=vec_d,
            z_j=self.z_j,
            vec_d_j=self.vec_d_j,
            T_hat_j=self.T_hat_j,
            alpha_j=self.alpha_j,
            sigma_par_j=self.sigma_par_j,
            sigma_perp_j=self.sigma_perp_j,
            quad_nodes=self.quad_nodes,
            eps_total=self.eps_total,
            n_chunk=self.n_chunk,
            m_chunk=self.m_chunk,
            dtype_override=self.c_dtype,
        )
