import torch.nn as nn

from typing import Optional


class ModulatedActivation(nn.Module):
    passive: bool = True
    count_modes: int = 7
    features: int = 1

    def __init__(
        self,
        passive: Optional[bool] = None,
        features: Optional[int] = None,
        count_modes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super(ModulatedActivation, self).__init__(**kwargs)

        # Main params.
        self.passive = passive if passive is not None else self.passive
        self.features = features if features is not None else self.features
        self.count_modes = count_modes if count_modes is not None else self.count_modes

        pass
