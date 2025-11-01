import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# --- Scaled Dot-Product Attention
# Q, K, V 모두 동일한 shape
# (batch_size, heads, seq_len, head_dim)
def attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(dim_k)
    
    # 디코더에서 사용될 Masking
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # (batch_size, heads, seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)

    return torch.matmul(weights, value)

# --- Token Embeddings + Positional Embeddings
class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 텐서를 GPU로 이동하고 모델의 속성으로 캐싱
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, inputs_ids):
        # inputs_ids: (batch_size, seq_len)
        seq_len = inputs_ids.size(1)
        
        # position_ids: (1, seq_len)
        position_ids = self.position_ids[:, :seq_len]
        
        token_embeddings = self.token_embeddings(inputs_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# --- Multi-Head-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_decoder=False):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 디코더에서 사용 중인지
        self.is_decoder = is_decoder 
        
        # Q, K, V를 한 번에 계산하기 위한 단일 Linear 레이어
        self.qkv_layer = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # 인코더-디코더 어텐션을 위한 K, V 레이어 (디코더에서 사용)
        if self.is_decoder:
            self.kv_layer = nn.Linear(self.embed_dim, 2 * self.embed_dim)

        self.output_layer = nn.Linear(self.embed_dim, self.embed_dim)

    def _split_heads(self, x, batch_size):
        # (batch_size, seq_len, embed_dim) 
        # -> (batch_size, num_heads, seq_len, head_dim)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_state, encoder_hidden_state=None, mask=None):
        batch_size = hidden_state.size(0)

        if encoder_hidden_state is None:
            # Self-Attention
            qkv = self.qkv_layer(hidden_state)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Cross-Attention
            q = self.qkv_layer(hidden_state).chunk(3, dim=-1)[0]
            kv = self.kv_layer(encoder_hidden_state)
            k, v = kv.chunk(2, dim=-1)
            
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        context = attention(q, k, v, mask)

        # 헤드 결합
        # 메모리 최적화를 위해 contignous 적용   
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embed_dim)
        
        output = self.output_layer(context)

        return output

# Position-wise Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x