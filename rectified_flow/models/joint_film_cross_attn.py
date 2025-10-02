import torch
import torch.nn as nn
import torch.nn.functional as F
from rectified_flow.models.joint_baseline import TimeEmbedding

class CrossAttention(nn.Module):
    """Minimal single-head cross-attention."""
    def __init__(self, dim_q, dim_kv):
        super().__init__()
        self.to_q = nn.Linear(dim_q, dim_q)
        self.to_k = nn.Linear(dim_kv, dim_q)
        self.to_v = nn.Linear(dim_kv, dim_q)
        self.proj = nn.Linear(dim_q, dim_q)

    def forward(self, x_q, x_kv):
        """
        x_q: (B, Nq, Dq)   e.g. image tokens
        x_kv: (B, Nk, Dkv) e.g. text tokens
        """
        Q = self.to_q(x_q)                     # (B, Nq, D)
        K = self.to_k(x_kv)                    # (B, Nk, D)
        V = self.to_v(x_kv)                    # (B, Nk, D)

        attn = torch.softmax(Q @ K.transpose(-2, -1) / (Q.size(-1)**0.5), dim=-1)
        out = attn @ V                         # (B, Nq, D)
        return self.proj(out)                  # (B, Nq, D)


class JointFiLMCrossAttn(nn.Module):
    def __init__(
        self, img_shape=(8, 8), img_dim=4, txt_dim=128,
        hidden=128, time_dim=128
    ):
        super().__init__()
        H, W = img_shape

        # --- image projection: keep spatial tokens ---
        self.img_proj = nn.Conv2d(img_dim, hidden, kernel_size=1)  # (B, hidden, H, W)
        self.img_norm = nn.LayerNorm(hidden)

        # --- text projection: still using LangVAE z (B, txt_dim) ---
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.txt_norm = nn.LayerNorm(hidden)

        # --- time embedding ---
        self.time_emb = TimeEmbedding(time_dim)
        self.t_proj = nn.Linear(time_dim, hidden)

        # --- FiLM conditioning ---
        self.film_gamma = nn.Linear(hidden + hidden, hidden)  # concat (t_emb, txt_vec)
        self.film_beta  = nn.Linear(hidden + hidden, hidden)

        # --- cross-attention ---
        self.cross_attn = CrossAttention(hidden, hidden)

        # --- token-wise feedforward ---
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden*2),
            nn.SiLU(),
            nn.Linear(hidden*2, hidden),
        )

        # --- heads ---
        self.img_head = nn.Conv2d(hidden, img_dim, kernel_size=1)
        self.txt_head = nn.Linear(hidden, txt_dim)

    def forward(self, x_img_t, x_txt_t, t):
        """
        x_img_t: (B, C=4, H=8, W=8)
        x_txt_t: (B, txt_dim)   (LangVAE latent)
        t: (B, 1)
        """
        B, C, H, W = x_img_t.shape

        # --- time embedding ---
        t_emb = self.t_proj(self.time_emb(t.view(B)))  # (B, hidden)

        # --- image tokens ---
        img_tok = self.img_proj(x_img_t)               # (B, hidden, H, W)
        img_tok = img_tok.permute(0, 2, 3, 1).reshape(B, H*W, -1)  # (B, 64, hidden)
        img_tok = self.img_norm(img_tok)

        # --- text token (we can treat as 1 token) ---
        txt_tok = self.txt_proj(x_txt_t).unsqueeze(1)  # (B, 1, hidden)
        txt_tok = self.txt_norm(txt_tok)

        # --- FiLM conditioning (time + text) ---
        cond = torch.cat([t_emb, txt_tok.squeeze(1)], dim=-1)   # (B, 2*hidden)
        gamma = self.film_gamma(cond).unsqueeze(1)              # (B,1,hidden)
        beta  = self.film_beta(cond).unsqueeze(1)
        img_tok = img_tok * (1 + gamma) + beta

        # --- cross-attention: img queries text ---
        img_tok = img_tok + self.cross_attn(img_tok, txt_tok)

        # --- feedforward per token ---
        img_tok = img_tok + self.ffn(img_tok)

        # --- outputs ---
        img_out = img_tok.reshape(B, H, W, -1).permute(0, 3, 1, 2)   # (B, hidden, H, W)
        v_img = self.img_head(img_out)                               # (B, C, H, W)

        # text head: pool text token after conditioning
        txt_feat = txt_tok.squeeze(1)
        u_txt = self.txt_head(txt_feat)                              # (B, txt_dim)

        return v_img, u_txt
