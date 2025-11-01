# Attention 메커니즘을 사용한 Transformer
Transformer 구조 학습 및 실습을 위한 구현

---

## 개요

- **인코더 및 디코더 구조**
  - Multi-Head Self-Attention와 Feed Forward Layer
  - Residual Connection과 Layer Normalization
  - Padding mask 지원

- **출력 (Output)**
  - hidden size → vocab size 변환 (`nn.Linear(hidden, vocab_size)`)
  - 활성화 함수로 softmax 사용

- **폴더 구조**
  ```
  Transformer/
  ├── main.ipynb            # main 파일 (테스트용)
  └── modules/
      ├── MyTransformer.py  # 인코더-디코더
      └── common.py         # 여러 번 재사용되는 구조들
                            ## attention, Embedding, MultiHeadAttention, FeedForward
  ```

- **사용 기술**
  - Python 3.12
  - PyTorch 2.9
  - NumPy

---

## 테스트 예시
### 0. 모델 import
```Python
import torch
from modules.MyTransformer import Transformer
```
### 1. 임의 파라미터 초기화
```
class Config:
    def __init__(self, vocab_size, pad_token_id=0):
        self.vocab_size = vocab_size        # 단어사전 크기
        self.hidden_size = 768              # 은닉층 차원수 (attention_eads의 배수)
        self.max_position_embeddings = 768  # 최대 시퀀스 길이
        self.num_attention_heads = 12       # 어텐션 헤드 개수
        self.num_hidden_layers = 12         # 레이어 개수
        self.intermediate_size = 2048       # FeedForward 레이어의 중간 차원
        self.hidden_dropout_prob = 0.1      # dropout 비율
        self.layer_norm_eps = 1e-12         # epsilon
        self.pad_token_id = pad_token_id    # 패딩 토큰
```

### 2. 더미 입력 데이터
```Python
source_ids = torch.tensor([
    [101, 2054, 2064, 2106, 102, 0, 0, 0, 0, 0, 0, 0],
    [101, 3000, 4000, 102, 2000, 102, 0, 0, 0, 0, 0, 0]
], dtype=torch.long)
```

### 3. 테스트 진행
```Python
model = Transformer(config_enc, config_dec)
model.eval() # 평가 모드

# ... (생략)

logits = model(source_ids, target_ids)
print(f"Output Logits shape: {logits.shape}")
```
---
## 문제 해결
### 1. RuntimeError: view size is not compatible with input tensor's size and stride
   
**상황**
- transpose를 적용한 텐서에 view를 사용할 때

**원인**
- expand, narrow, transpose 등 배열의 형태를 바꾸는 메서드를 적용하면 메모리의 stride가 바뀐다.
- 배열의 저장 형태가 non-unit stride일 경우 view가 읽어오지 못한다.

**해결**
```Python
context = context.transpose(1, 2).contiguous()
context = context.view(batch_size, -1, self.embed_dim)
```
contiguous()를 추가로 적용


### 2. AttributeError: 'Embedding' object has no attribute 'self.position_ids'

**상황**
- Embedding 클래스가 호출될 때 `self.position_ids` 속성에 접근하지 못함

**원인**
- 파라미터에 device 속성을 적용
- 다른 파라미터들은 GPU로 이동했는데 텐서만 여전히 CPU에 남아있어서 서로 connected되지 못함
- torch.tensor()는 model.to()가 적용되지 않음

**해결**
```Python
# torch.tensor는 GPU로 이동하기 위해 별도의 명시가 필요
self.register_buffer("position_ids", torch.arange(...))
```
