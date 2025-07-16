import torch
from torch import nn



class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()


    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            loss (Tensor): calculated loss value.
        """
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        return self.loss(logits, labels)
