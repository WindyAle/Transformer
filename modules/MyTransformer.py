import torch
import torch.nn as nn

from modules.common import FeedForward, Embedding, MultiHeadAttention

# --- Encoder
# 하나의 레이어는 Self-Attention과 FeedForward를 수행
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.attention = MultiHeadAttention(config, is_decoder=False)
        self.feed_forward = FeedForward(config)

    def forward(self, x, mask=None):
        # 1. Self-Attention
        hidden_state_norm = self.layer_norm1(x)
        # Multi-Head-Attention일 경우 mask 전달
        attention_output = self.attention(hidden_state_norm, mask=mask)
        x = x + attention_output
        
        # 2. FeedForward
        hidden_state_norm = self.layer_norm2(x)
        ff_output = self.feed_forward(hidden_state_norm)
        x = x + ff_output

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()

        # 임베딩 레이어를 외부에서 받거나(공유) 자체 생성
        self.embeddings = embedding if embedding is not None else Embedding(config)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layer_norm_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs_ids, attention_mask=None):
        hidden_state = self.embeddings(inputs_ids)
        
        for layer in self.layers:
            hidden_state = layer(hidden_state, mask=attention_mask)
            
        hidden_state = self.layer_norm_final(hidden_state)

        return hidden_state

# --- Decoder
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(config, is_decoder=False) 
        # Cross Attention
        self.cross_attention = MultiHeadAttention(config, is_decoder=True)
        # FeedForward
        self.feed_forward = FeedForward(config)

    def forward(self, x, encoder_hidden_state, self_attn_mask=None, cross_attn_mask=None):
        # 1. Masked Self-Attention
        norm_x = self.layer_norm1(x)
        attn_output = self.self_attention(norm_x, 
                                          encoder_hidden_state=None, 
                                          mask=self_attn_mask)
        x = x + attn_output
        
        # 2. Cross-Attention
        norm_x = self.layer_norm2(x)
        attn_output = self.cross_attention(norm_x, 
                                           encoder_hidden_state=encoder_hidden_state, 
                                           mask=cross_attn_mask)
        x = x + attn_output
        
        # 3. FeedForward
        norm_x = self.layer_norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()

        self.embeddings = embedding if embedding is not None else Embedding(config)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.layer_norm_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, target_ids, encoder_hidden_state, self_attn_mask=None, cross_attn_mask=None):       
        hidden_state = self.embeddings(target_ids)
        
        for layer in self.layers:
            hidden_state = layer(
                x=hidden_state,
                encoder_hidden_state=encoder_hidden_state,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask
            )
            
        hidden_state = self.layer_norm_final(hidden_state)

        return hidden_state

# --- 최종 모델
# 인코더와 디코더 연결
class Transformer(nn.Module):
    def __init__(self, config_encoder, config_decoder):
        super().__init__()
        
        # 임베딩 레이어 정의
        # 소스 언어와 타겟 언어의 단어사전이 다를 수 있으므로 별도 생성
        self.encoder_embeddings = Embedding(config_encoder)
        self.decoder_embeddings = Embedding(config_decoder)
        
        # 1. 인코더 + 디코더
        self.encoder = TransformerEncoder(config_encoder, self.encoder_embeddings)
        self.decoder = TransformerDecoder(config_decoder, self.decoder_embeddings)
        
        # 2. 최종 출력 선형 레이어
        self.output_layer = nn.Linear(config_decoder.hidden_size, config_decoder.vocab_size)
        
        self.pad_token_id = config_encoder.pad_token_id

    def create_padding_mask(self, input_ids):
        # (batch_size, 1, 1, seq_len)
        mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask

    def create_look_ahead_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        # (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, source_ids, target_ids):
        device = source_ids.device
        target_seq_len = target_ids.size(1)
        
        # 1. 마스크 생성
        # 소스 패딩 (Encoder Self-Attention + Cross-Attention에서 사용)
        # (batch_size, 1, 1, source_seq_len)
        source_mask = self.create_padding_mask(source_ids)
        
        # 타겟 패딩
        # (batch_size, 1, 1, target_seq_len)
        target_mask = self.create_padding_mask(target_ids)
        
        # 룩 어헤드 마스킹 (Decoder용)
        # (1, 1, target_seq_len, target_seq_len)
        look_ahead_mask = self.create_look_ahead_mask(target_seq_len, device)
        
        # 최종 디코더 셀프 어텐션 마스크 (패딩 + 룩 어헤드)
        # (batch_size, 1, target_seq_len, target_seq_len)
        decoder_self_attn_mask = target_mask & look_ahead_mask
        
        # 2. 인코더
        # (batch_size, source_seq_len, hidden_size)
        encoder_output = self.encoder(source_ids, attention_mask=source_mask)
        
        # 3. 디코더
        # (batch_size, target_seq_len, hidden_size)
        decoder_output = self.decoder(
            target_ids=target_ids,
            encoder_hidden_state=encoder_output,
            self_attn_mask=decoder_self_attn_mask,
            cross_attn_mask=source_mask # Cross-Attention에는 소스 마스크 사용
        )
        
        # 4. 최종 출력
        # (batch_size, target_seq_len, target_vocab_size)
        logits = self.output_layer(decoder_output)
        
        return logits