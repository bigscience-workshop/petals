import torch
import torch.nn as nn
from hivemind.moe.server.layers.custom_experts import register_expert_class


@register_expert_class("ExampleModule", lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)))
class ExampleModule(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.ffn = nn.Linear(hid_dim, 4 * hid_dim)
        self.ffn_output = nn.Linear(4 * hid_dim, hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-12)

    def forward(self, x):
        ffn_output = self.ffn(x)
        ffn_output = torch.nn.functional.gelu(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return self.layer_norm(x + ffn_output)
