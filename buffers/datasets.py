import torch


class CustomDataset:
    def __init__(self, dataset_size: int, device: torch.device):
        super().__init__()
        self.dataset_size = dataset_size
        self.data = torch.tensor([], device=device)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(
        self, idx: torch.Tensor | tuple[int | slice | torch.Tensor, int | slice | torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(idx, tuple):
            assert len(idx) == 1
        return self.data[idx]

    def add(self, new_batch: torch.Tensor) -> None:
        if self.data.shape[0] == 0:
            self.data = self.data.to(new_batch.dtype)
        self.data = torch.cat([self.data, new_batch.detach()], dim=0)
        self.trim_if_needed()

    def trim_if_needed(self) -> None:
        if len(self) > self.dataset_size:
            trim_from = self.data.shape[0] - self.dataset_size
            self.data = self.data[trim_from:]  # FIFO

    def update(
        self,
        indices: torch.Tensor | tuple[int | slice | torch.Tensor, int | slice | torch.Tensor],
        new_batch: torch.Tensor,
    ) -> None:
        if isinstance(indices, tuple):
            assert len(indices) == 1
        self.data[indices] = new_batch.detach()

    def discard(self, indices: torch.Tensor) -> None:
        assert indices.ndim == 1
        self.data = self.data[~indices]
