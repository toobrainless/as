import torch
from torch import Tensor, nn


class CTCLossWrapper(nn.modules.loss._Loss):
    def __init__(self, blank=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank)

    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        return self.ctc_loss(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
