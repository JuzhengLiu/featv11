import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE_Layer(nn.Module):
    """
    MoE Layer with Zero-Init for Residual Learning
    """
    def __init__(self, input_dim, output_dim, num_experts=4, k=2, 
                 noisy_gating=False):
        super(MoE_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        # Router
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 4),
            nn.Tanh(),
            nn.Linear(num_experts * 4, num_experts)
        )
        
        # Experts: MLP
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim)
            )
            # Zero-Init Last Layer
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
            self.experts.append(expert)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Router
        logits = self.gate(x) 
        if self.training and self.noisy_gating:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        scores = F.softmax(logits, dim=-1) 
        
        # Top-K
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Aux Loss
        if self.training:
            importance = scores.sum(dim=0)
            mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
            load = mask.sum(dim=0)
            aux_loss = (self.num_experts * (importance * load).sum()) / (batch_size * batch_size)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        # Compute
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        for i in range(self.k):
            idx = topk_indices[:, i] 
            weight = topk_scores[:, i].unsqueeze(1) 
            for expert_idx in range(self.num_experts):
                mask = (idx == expert_idx)
                if mask.sum() > 0:
                    output[mask] += weight[mask] * self.experts[expert_idx](x[mask])
                    
        return output, aux_loss