import torch


from dyna.lib.cpsf.structures import CPSFLatticeSumPolicy


def fixed_window(
    policy: CPSFLatticeSumPolicy,
) -> torch.LongTensor:
    raise NotImplementedError
