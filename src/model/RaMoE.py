import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopKGating(nn.Module):
    def __init__(self, hid_dim: int, n_exps: int, noisy: bool = True, noise_eps: float = 1e-2):
        super().__init__()
        self.noisy = noisy
        self.noise_eps = noise_eps
        self.w_gate  = nn.Parameter(torch.zeros(hid_dim, n_exps))
        self.w_noise = nn.Parameter(torch.zeros(hid_dim, n_exps))
        nn.init.normal_(self.w_gate,  mean=0.0, std=0.02)
        nn.init.normal_(self.w_noise, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, r: torch.Tensor | None = None, training: bool = False) -> torch.Tensor:
        clean_logits = x @ self.w_gate  # (B, E)
        if self.noisy and training:
            raw_noise_std = x @ self.w_noise
            noise_std = F.softplus(raw_noise_std) + self.noise_eps
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits

        if r is not None:
            temp = torch.sigmoid(r).clamp_min(1e-6)
            logits = logits / temp

        gates = F.softmax(logits, dim=-1)
        return gates

class CrossAttnExpert(nn.Module):
    def __init__(self, hid_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(hid_dim) 
        self.attn   = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_heads,
                                            dropout=dropout, batch_first=True)
        self.drop1  = nn.Dropout(dropout)

        self.norm_o = nn.LayerNorm(hid_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(hid_dim, ffn_mult * hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * hid_dim, hid_dim),
            nn.Dropout(dropout),
        )

    @torch.no_grad()
    def _shape_check(self, query: torch.Tensor, retrieved: torch.Tensor):
        assert retrieved.dim() == 3, "retrieved must be (B, L, D)"
        if query.dim() == 2:
            return
        if query.dim() == 3 and query.size(1) == 1:
            return
        raise ValueError("query must be (B, D) or (B, 1, D)")

    def forward(self,
                query: torch.Tensor,
                retrieved: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        self._shape_check(query, retrieved)

        if query.dim() == 2:
            query = query.unsqueeze(1)

        q = self.norm_q(query)
        kv = retrieved         

        attn_out, attn_probs  = self.attn(q, kv, kv,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True,
                                average_attn_weights=True)

        x = query + self.drop1(attn_out)

        y = self.ffn(self.norm_o(x)) + x 
        return y.squeeze(1), attn_probs.squeeze(1)

class MoEAttentionPerExpert(nn.Module):
    def __init__(self, 
            hid_dim: int, 
            n_exps: int, 
            num_heads: int = 8, 
            dropout: float = 0.1, 
            alpha: float = 1.0,
            noisy: bool = True, 
            ffn_mult: int = 4, 
            topk: int | None = None,
            beta: float = 0.5 
        ):
        super().__init__()
        self.n_exps = n_exps
        self.alpha  = alpha
        self.topk   = topk
        self.beta   = beta
        self.gating = NoisyTopKGating(hid_dim, n_exps, noisy=noisy)
        self.experts = nn.ModuleList([
            CrossAttnExpert(hid_dim, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(n_exps)
        ])

    def _entropy(self, p: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        p_safe = p.clamp_min(eps)
        return -(p_safe * p_safe.log()).sum(dim=dim)

    def forward(self,
                query: torch.Tensor,
                retrieved: torch.Tensor,
                r: torch.Tensor | None = None,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = query.shape
        gates = self.gating(query, r=r, training=False) 

        expert_outs = []
        reliabilities = [] 
        if self.topk is None or self.topk >= self.n_exps:
            expert_outs = []
            for e in self.experts:
                out_e, attn_prob = e(query, retrieved, attn_mask=attn_mask, key_padding_mask=key_padding_mask) 
                expert_outs.append(out_e.unsqueeze(1))
                H = self._entropy(attn_prob, dim=-1)                  
                s = torch.exp(-self.beta * H) 
                reliabilities.append(s.unsqueeze(1))               
            expert_outs = torch.cat(expert_outs, dim=1)
            s_all = torch.cat(reliabilities, dim=1)
        else:
            topk_vals, topk_idx = torch.topk(gates, k=self.topk, dim=-1)
            mask = torch.zeros_like(gates).scatter(1, topk_idx, 1.0)
            gates = gates * mask
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-9)

            expert_outs = torch.zeros(B, self.n_exps, D, device=query.device, dtype=query.dtype)
            s_all       = torch.zeros(B, self.n_exps, device=query.device, dtype=query.dtype)

            for i in range(self.topk):
                idx = topk_idx[:, i]
                for e_id in idx.unique():
                    sel = (idx == e_id)
                    if sel.any():
                        q_sub = query[sel]
                        r_sub = retrieved[sel]
                        out_i, attn_prob = self.experts[int(e_id)](q_sub, r_sub,
                                                                  attn_mask=None, key_padding_mask=None)
                        expert_outs[sel, int(e_id)] = out_i
                        H = self._entropy(attn_prob, dim=-1)   
                        s = torch.exp(-self.beta * H)  
                        s_all[sel, int(e_id)] = s

        gates_tilde = gates * s_all                              
        gates_tilde = gates_tilde / (gates_tilde.sum(dim=1, keepdim=True) + 1e-9)

        mixture = (gates_tilde.unsqueeze(-1) * expert_outs).sum(dim=1)
        fused   = self.alpha * mixture + (1.0 - self.alpha) * query

        return fused, expert_outs, gates_tilde

class ModalFusionLayer(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 multi: int,          
                 num_modalities: int,
                 dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.num_modalities = num_modalities

        self.modal_expert_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU()
                )
                for _ in range(multi)
            ])
            for _ in range(num_modalities)
        ])
        self.ent_attn = nn.Linear(out_dim, 1, bias=False)

    def forward(self, *modal_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(modal_embs) == self.num_modalities, \
            f"Expected {self.num_modalities} modal embeddings, got {len(modal_embs)}"

        fused_results = []
        attn_weights = None

        for i in range(self.multi):
            proj_embs = []
            for m_idx in range(self.num_modalities):
                x_proj = self.modal_expert_layers[m_idx][i](modal_embs[m_idx]) 
                proj_embs.append(x_proj)

            x_stack = torch.stack(proj_embs, dim=1)

            attn_scores = self.ent_attn(x_stack).squeeze(-1) 
            attn_weights = torch.softmax(attn_scores, dim=-1) 

            context_vector = torch.sum(attn_weights.unsqueeze(-1) * x_stack, dim=1)
            fused_results.append(context_vector)

        fused_results = torch.stack(fused_results, dim=1)
        fused_output = fused_results.sum(dim=1) 

        return fused_output, attn_weights

    def gated_fusion(self, emb: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel

class RaMoE(nn.Module):
    def __init__(self,
                 modalities: tuple[str, ...] = ("text", "image"),
                 retrieval_num: int = 500,  
                 num_experts: int = 3,
                 fea_dim: int = 768,
                 dropout: float = 0.2,
                 num_head: int = 8,
                 alpha: float = 0.5,
                 delta: float = 0.25,
                 num_epoch: int = 20,
                 **kwargs):
        super().__init__()
        assert len(modalities) >= 2, "At least two modalities are required."
        self.modalities = modalities
        self.num_modalities = len(modalities)

        self.expert_modules = nn.ModuleDict({
            mod: MoEAttentionPerExpert(
                hid_dim=fea_dim,
                n_exps=num_experts,
                dropout=dropout,
                num_heads=num_head,
                alpha=alpha
            )
            for mod in modalities
        })

        self.fuse_layer = ModalFusionLayer(
            in_dim=fea_dim,
            out_dim=fea_dim,
            multi=2,
            num_modalities=len(self.modalities),
            dropout=dropout
        )

        self.predictors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.LazyLinear(fea_dim),
                nn.ReLU(),
                nn.LazyLinear(1)
            )
            for mod in modalities
        })

        self.mixture_predictor = nn.Sequential(
            nn.LazyLinear(fea_dim),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

        self.label_embedding_linear = nn.Sequential(
            nn.LazyLinear(fea_dim),
            nn.Sigmoid()
        )

        self.delta = delta
        self.total_epoch = num_epoch

    def forward(self,
                inputs: dict,
                retrieved_item_similarity: torch.Tensor,
                retrieved_label_list: torch.Tensor) -> dict:
        modal_embs: dict[str, torch.Tensor] = {}
        modal_attn: dict[str, torch.Tensor] = {}
        modal_disen: dict[str, torch.Tensor] = {}

        similarity = torch.softmax(retrieved_item_similarity, dim=1) 
        weighted_label = similarity * retrieved_label_list            
        label_embedding = self.label_embedding_linear(weighted_label) 
        label_embedding = label_embedding.unsqueeze(1)               
        enhanced_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for mod in self.modalities:
            q, retrieved = inputs[mod]                           
            enhanced_retrieved = retrieved * label_embedding
            enhanced_inputs[mod] = (q, enhanced_retrieved)

        for mod in self.modalities:
            q, retrieved = enhanced_inputs[mod]
            emb, expert_outs, gates = self.expert_modules[mod](q, retrieved)
            modal_embs[mod] = emb
            modal_attn[mod] = gates           
            modal_disen[mod] = expert_outs 

        emb_list = [modal_embs[m] for m in self.modalities]
        fused, fusion_attn = self.fuse_layer(*emb_list)  

        modal_preds = {m: self.predictors[m](modal_embs[m]) for m in self.modalities} 
        mm_pred = self.mixture_predictor(fused)

        return {
            'mixture_prediction': mm_pred,
            'fusion_attention': fusion_attn,
            'modal_predictions': modal_preds,
            'modal_attentions': modal_attn,
            'modal_disentanglements': modal_disen
        }

    def calculate_loss(self, **inputs) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.delta
        total_epoch = self.total_epoch

        label = inputs['label'].view(-1, 1)
        mixture_prediction = inputs['mixture_prediction']
        modal_predictions = inputs.get('modal_predictions', None)
        current_epoch = inputs['epoch']

        epoch_factor = (float(current_epoch) / float(total_epoch)) ** 2
        epoch_factor = min(max(epoch_factor, 0.0), 1.0)
        mixture_loss = F.mse_loss(mixture_prediction, label)

        if modal_predictions is not None and len(modal_predictions) > 0:
            modal_losses = [F.mse_loss(pred, label) for pred in modal_predictions.values()]
            experiment_loss = sum(modal_losses) / len(modal_losses)
        else:
            experiment_loss = torch.tensor(0.0, device=mixture_prediction.device)

        w_modal = (1 - epoch_factor) * delta
        w_mixture = 1.0 - w_modal

        if torch.isnan(experiment_loss):
            experiment_loss = torch.tensor(0.0, device=mixture_loss.device)
        if torch.isnan(mixture_loss):
            mixture_loss = torch.tensor(0.0, device=mixture_loss.device)

        joint_loss = w_modal * experiment_loss + w_mixture * mixture_loss
        return joint_loss, mixture_loss
