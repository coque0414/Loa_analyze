from transformers import AutoTokenizer, AutoModel
import torch

# 1. 모델과 토크나이저 불러오기
model_name = "BM-K/KoSimCSE-roberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 테스트 문장
sentence = "낙원 콘텐츠"

# 3. 토큰화 (입력 텐서로 변환)
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# 4. 모델에 입력하여 hidden states 얻기
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
    attention_mask = inputs['attention_mask']

# 5. Sentence-Embedding: [CLS] 벡터 or Mean Pooling
# 👉 KoSimCSE는 일반적으로 Mean Pooling 사용
sentence_embedding = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

print(sentence_embedding.shape)  # (1, 768)
print(sentence_embedding)
