import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from sys import exit
import math
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, mlp_bias = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 2
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias,dtype=torch.bfloat16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias,dtype=torch.bfloat16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias,dtype=torch.bfloat16)


    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.silu(gate)
        up = self.up_proj(x)
        gated_output = gate * up
        down_proj = self.down_proj(gated_output)
        return down_proj

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim = False):
        super().__init__()
        self.head_dim = head_dim
        if self.head_dim == False:
            self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_heads, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_heads, bias=False)
        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.embed_dim, bias=False)
    def forward(self, hidden_states, R_matrix, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        attn_output, attn_weights =  self.attention(query_states, key_states, value_states,R_matrix, attention_mask)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


    def attention(self, query_states, key_states, value_states, R_matrix, attention_mask):
        query_states_reordered = query_states.permute(0, 2, 1, 3).reshape(query_states.shape[0], query_states.shape[2], -1)
        key_states_reordered = key_states.permute(0, 2, 1, 3).reshape(key_states.shape[0], key_states.shape[2], -1)
        #query_rotated = torch.einsum("bte,ted->btd", query_states_reordered, R_matrix[:query_states.shape[2], :, :])
        #key_rotated = torch.einsum("bse,sed->bsd", key_states_reordered, R_matrix[:key_states.shape[2], :, :])

        #query_rotated = query_rotated.reshape(query_states.shape[0], query_states.shape[2], query_states.shape[1], query_states.shape[3]).permute(0, 2, 1, 3)
        #key_rotated = key_rotated.reshape(key_states.shape[0], key_states.shape[2], key_states.shape[1], key_states.shape[3]).permute(0, 2, 1, 3)



        R_matrix = R_matrix.unsqueeze(0).expand(query_states_reordered.size(0), -1, -1, -1)
        query_rotated = query_states_reordered.unsqueeze(2)
        query_rotated = torch.matmul(query_rotated, R_matrix)
        query_rotated = query_rotated.reshape(query_states.shape[0], query_states.shape[2], query_states.shape[1], query_states.shape[3]).permute(0, 2, 1, 3)
        key_rotated = key_states_reordered.unsqueeze(2)
        key_rotated = torch.matmul(key_rotated, R_matrix)
        key_rotated = key_rotated.reshape(key_states.shape[0], key_states.shape[2], key_states.shape[1], key_states.shape[3]).permute(0, 2, 1, 3)



        attn_weights = torch.matmul(query_rotated, key_rotated.transpose(-2, -1)) * self.scaling

        #attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        #self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        #self.attn = MultiheadAttention(embed_dim, num_heads, head_dim)
        self.norm1 = RMSNorm(embed_dim)
        self.MLP = SwiGLU(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, head_dim * num_heads, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(embed_dim, head_dim * num_heads, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(embed_dim, head_dim * num_heads, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(head_dim * num_heads, embed_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x, cos, sin, device):
        #x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        #attn_out, _ = self.attn(x,x,x)
        #attn_out, _ = self.attn(x, R_matrix, attention_mask=mask)
        x = x.to(torch.float32)
        x = self.norm1(x)
        x = x.to(torch.bfloat16)
        attn_out = self.attention(x, cos, sin, device)
        x = x + attn_out
        x = x.to(torch.float32)
        x = self.norm2(x)
        x = x.to(torch.bfloat16)
        ff_out = self.MLP(x)
        x = x + ff_out
        return x


    def attention(self, x, cos, sin, device):
        #x.to(torch.float32)
        x = x.to(torch.bfloat16)
        batch_size, seq_len, _ = x.shape
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        #q = self.q_proj(x).view(*hidden_shape).to(torch.float16)
        q = self.q_proj(x).view(*hidden_shape)
        q_rotated = self.apply_rotary_embedding(q, cos, sin, device)

        k = self.k_proj(x).view(*hidden_shape)
        k_rotated = self.apply_rotary_embedding(k, cos, sin, device)


        v = self.v_proj(x).view(*hidden_shape)

        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            print("NaN qkv in attention inputs!")

        attn_output = flash_attn_func(q_rotated, k_rotated,v, dropout_p=0.1,causal=True)
        if torch.isnan(attn_output).any():
            print("NaN detected in attention inputs!")
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


    def apply_rotary_embedding(self, x, cos, sin, device):
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        seq_len = x.shape[1]

        cos = cos[:, :seq_len, :]
        sin = sin[:, :seq_len, :]
        x1 = x[..., ::2].to(device)
        x2= x[..., 1::2].to(device)


        x_rot_even = x1 * cos - x2 * sin
        x_rot_odd = x1 * sin + x2 * cos

        x_rot = torch.empty_like(x).to(device)
        x_rot[..., ::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot



class Llama(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, head_dim, max_len, num_layers=1, device="cuda"):
        super().__init__()
        self.max_len = max_len - 1
        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head_dim = head_dim
        #self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.cos, self.sin = self.get_rotary_angles(self.max_len, self.head_dim, device=self.device)
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(embed_dim, num_heads, head_dim=head_dim) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size,dtype=torch.bfloat16)


        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)




    def forward(self, x, labels=None):
        B, T = x.shape
        x = self.embed(x)
        #x = self.embed(x) + self.pos_embed[:, :T, :]
        #mask = torch.triu(torch.ones(T, T), diagonal=1)

        for block in self.decoder_blocks:
            x = block(x, self.cos, self.sin, self.device)

        logits = self.fc_out(x)  # [B, T, vocab_size]

        if labels is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss

        return logits


    def get_rotary_angles(self, seqlen, headdim, base=10000, device='cpu', dtype=torch.bfloat16):

        dim_half = headdim // 2
        theta = torch.arange(dim_half, device=device, dtype=dtype)
        theta = 1.0 / (base ** (theta / dim_half))

        positions = torch.arange(seqlen, device=device, dtype=dtype)

        angles = torch.einsum("i,j->ij", positions, theta)  # (seqlen, headdim//2)

        cos = angles.cos()
        sin = angles.sin()
        return cos, sin

