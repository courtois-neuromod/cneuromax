""":class:`OneKLKWavDataset`."""
from torch import Tensor

from cneuromax.projects.haptic.klk_wav import KLKWavDataset


class OneKLKWavDataset(KLKWavDataset):
    """:mod:`one_klk_wav`` :class:`.KLKWavDataset`."""

    def __len__(self: "OneKLKWavDataset") -> int:
        """See :meth:`torch.utils.data.Dataset.__len__`."""
        return 1

    def __getitem__(
        self: "OneKLKWavDataset",
        idx: int,
    ) -> dict[str, Tensor]:
        """See :meth:`torch.utils.data.Dataset.__getitem__`."""
        return {"BR": self.dataset[self.dataset_XXXX_list[0]]["BR"]}
