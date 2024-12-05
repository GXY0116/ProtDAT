import math
import torch
import torch.nn as nn
import torch.nn.functional as F  


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    attention_scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(q.shape[-1])
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, value=float('-inf'))
    attention_weights = F.softmax(attention_scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)
       
    output = torch.matmul(attention_weights, v)
    return output


def precompute_freqs_cis(dim, end, theta, device):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).to(device)
    t = torch.arange(end, dtype=torch.float32).to(device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, freqs_cis):
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    freqs_cis_q = reshape_for_broadcast(freqs_cis, q_)
    q_out = torch.view_as_real(q_ * freqs_cis_q).flatten(3)
    
    return q_out.type_as(q)


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(F.silu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
 
    
class MCM(torch.nn.Module):
    def __init__(self, n_heads, d_model, cross_vocab_size, dropout=0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.cross_vocab_size = cross_vocab_size
        self.d_emb = d_model//self.n_heads
        
        assert (
            self.d_emb * self.n_heads == self.d_model
        ), "Embedding size needs to be divisible by heads"
        
        self.ln_des = nn.LayerNorm(self.d_model)
        self.ln_des_2 = nn.LayerNorm(self.d_model)
        self.ln_cross = nn.LayerNorm(self.d_model)
        self.ln_cross_2 = nn.LayerNorm(self.d_model)
        self.ln_seq = nn.LayerNorm(self.d_model)
        self.ln_seq_2 = nn.LayerNorm(self.d_model)
        
        self.des_q_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.des_k_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.des_v_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.des_out = torch.nn.Linear(self.n_heads*self.d_emb, d_model)
        
        self.cross_q_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.cross_k_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.cross_v_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.cross_out = torch.nn.Linear(self.n_heads*self.d_emb, d_model)
        
        self.seq_q_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.seq_k_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.seq_v_linear_layer = nn.Linear(self.d_model, d_model, bias=False)
        self.seq_out = torch.nn.Linear(self.n_heads*self.d_emb, d_model)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.feed_d = FeedForward(d_model, d_model*4, dropout=0.1)
        self.feed_c = FeedForward(d_model, d_model*4, dropout=0.1)
        self.feed_s = FeedForward(d_model, d_model*4, dropout=0.1)
        
    def forward(self, des_vec, cross_vec, seq_vec, mask, freqs_cis):
        b_s, n_d, n_c, n_s = des_vec.shape[0], des_vec.shape[1], cross_vec.shape[1], seq_vec.shape[1]
        
        residual_des, residual_cross, residual_seq = des_vec, cross_vec, seq_vec
        des_vec, cross_vec, seq_vec = self.ln_des(des_vec), self.ln_cross(cross_vec), self.ln_seq(seq_vec)
        
        des_mask, cross_mask_des, seq_mask = mask[0], mask[1], mask[2]
        
        des_new_q = self.des_q_linear_layer(des_vec).view(b_s, n_d, self.n_heads, self.d_emb)
        des_new_k = self.des_k_linear_layer(des_vec).view(b_s, n_d, self.n_heads, self.d_emb)
        des_new_v = self.des_v_linear_layer(des_vec).view(b_s, n_d, self.n_heads, self.d_emb).permute(0, 2, 1, 3)
        
        cross_pro_q = self.cross_q_linear_layer(cross_vec).view(b_s, n_c, self.n_heads, self.d_emb)
        
        cross_pro_q, des_new_q, des_new_k = apply_rotary_emb(cross_pro_q, freqs_cis[:n_c]).permute(0, 2, 1, 3), apply_rotary_emb(des_new_q, freqs_cis[:n_d]).permute(0, 2, 1, 3), apply_rotary_emb(des_new_k, freqs_cis[:n_d]).permute(0, 2, 1, 3)
        
        des_output = scaled_dot_product_attention(q=des_new_q, k=des_new_k, v=des_new_v, mask=des_mask, dropout=self.dropout)
        des_output = self.des_out(des_output.permute(0, 2, 1, 3).contiguous().view(b_s, n_d, self.n_heads*self.d_emb))
        cross_output = scaled_dot_product_attention(q=cross_pro_q, k=des_new_k, v=des_new_v, mask=cross_mask_des, dropout=self.dropout)
        cross_output = self.cross_out(cross_output.permute(0, 2, 1, 3).contiguous().view(b_s, n_c, self.n_heads*self.d_emb))
        
        des_output, cross_output = des_output + residual_des, cross_output + residual_cross
        residual_des, residual_cross = des_output, cross_output
        des_output, cross_output = self.ln_des_2(des_output), self.ln_cross_2(cross_output)

        seq_new_q = self.seq_q_linear_layer(seq_vec).view(b_s, n_s, self.n_heads, self.d_emb)
        seq_new_k = self.seq_k_linear_layer(seq_vec).view(b_s, n_s, self.n_heads, self.d_emb)
        seq_new_v = self.seq_v_linear_layer(seq_vec).view(b_s, n_s, self.n_heads, self.d_emb)
        
        cross_pro_k = self.cross_k_linear_layer(cross_output).view(b_s, n_c, self.n_heads, self.d_emb)
        cross_pro_v = self.cross_v_linear_layer(cross_output).view(b_s, n_c, self.n_heads, self.d_emb)
        
        seq_new_q, seq_new_k, cross_pro_k = apply_rotary_emb(seq_new_q, freqs_cis[:n_s]).permute(0, 2, 1, 3), apply_rotary_emb(seq_new_k, freqs_cis[:n_s]), apply_rotary_emb(cross_pro_k, freqs_cis[:n_c])

        cross_des_c_k = torch.cat((cross_pro_k, seq_new_k), 1).permute(0, 2, 1, 3)
        cross_des_c_v = torch.cat((cross_pro_v, seq_new_v), 1).permute(0, 2, 1, 3)
        
        seq_output = scaled_dot_product_attention(q=seq_new_q, k=cross_des_c_k, v=cross_des_c_v, mask=seq_mask, dropout=self.dropout)
        seq_output = self.seq_out(seq_output.permute(0, 2, 1, 3).contiguous().view(b_s, n_s, self.n_heads*self.d_emb))
        
        seq_output = seq_output + residual_seq
        residual_seq = seq_output
        seq_output = self.ln_seq_2(seq_output)
        
        des_output = self.feed_d(des_output)
        cross_output = self.feed_c(cross_output)
        seq_output = self.feed_s(seq_output)

        des_output = self.dropout(des_output + residual_des)
        cross_output = self.dropout(cross_output + residual_cross)
        seq_output = self.dropout(seq_output + residual_seq)
        
        return des_output, cross_output, seq_output
    

class ProtDAT_Decoder(torch.nn.Module):
    def __init__(self, device, d_model, des_vocab_size, seq_vocab_size, cross_vocab_size, layer, head_num, dropout):
        super().__init__()

        self.d_model = d_model
        self.seq_emb = nn.Embedding(seq_vocab_size, d_model)
        self.cross_emb = nn.Embedding(cross_vocab_size, d_model)
        self.cross_vocab_size = cross_vocab_size
        self.head_num = head_num
        self.device = device
        
        self.blocks = nn.ModuleList([MCM(n_heads=self.head_num, d_model=self.d_model, cross_vocab_size=self.cross_vocab_size, dropout=0.1) for _ in range(layer)])
        
        self.emb2seq = nn.Linear(self.d_model, seq_vocab_size)
        self.emb2descrip = nn.Linear(self.d_model, des_vocab_size)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.ln_des, self.ln_seq, self.ln_cross = nn.LayerNorm(self.d_model), nn.LayerNorm(self.d_model), nn.LayerNorm(self.d_model)
        self.freqs_cis = precompute_freqs_cis(dim=self.d_model//self.head_num, end=2048, theta=10000.0, device=device).to(self.device)
        
    def forward(self, des=None, seq=None, mask=None):
        cross = self.cross_emb(torch.arange(self.cross_vocab_size).repeat(des.shape[0], 1).to(self.device))
        
        d_emb = des
        s_emb = self.seq_emb(seq)
        
        self.dropout(d_emb)
        self.dropout(s_emb)
        
        self.des_attention_weights = []
        self.cross_attention_weights = []
        self.seq_attention_weights = []
        
        for block in self.blocks:
            d_emb, cross, s_emb = block(des_vec=d_emb, cross_vec=cross, seq_vec=s_emb, mask=mask, freqs_cis=self.freqs_cis)
        
        d_emb, cross, s_emb = self.ln_des(d_emb), self.ln_cross(cross), self.ln_seq(s_emb)
        
        return d_emb, cross, s_emb