import torch
import torch.nn as nn
import math
from utils import *
MAX_LEN = 128

### ==================================================
### (1) TO-DO: Model Definition
### ==================================================
### Base transformer layers in "Attention Is All You Need"
###   TransformerEncoderLayer
###   TransformerDecoderLayer
###   Positional encoding and input embedding
###   Note that you may need masks when implementing attention mechanism
###     Padding mask: prevent input from attending to padding tokens
###     Causal mask: prevent decoder input from attending to future input

class MultiHeadAttention(nn.Module):
    """
    多頭注意力機制。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每個 head 的維度
        
        # Q, K, V 和輸出的線性層
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # q, k, v shape: (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # 1. Linear 投影並切分 head
        # (B, L, D) -> (B, L, H, D_k) -> (B, H, L, D_k)
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # (B, H, L_q, D_k) @ (B, H, D_k, L_k) -> (B, H, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 套用 mask (padding mask or causal mask)
        # mask shape: (B, 1, L_q, L_k) or (B, 1, 1, L_k) or (1, 1, L_q, L_q)
        if mask is not None:
            # 將 mask 為 0 (False) 的地方填上 -1e9 (一個很小的數)，softmax 後會趨近 0
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        # 4. Attention 結果與 V 相乘
        # (B, H, L_q, L_k) @ (B, H, L_v, D_k) -> (B, H, L_q, D_k) (L_k == L_v)
        context = torch.matmul(attn, v)
        
        # 5. 合併 heads 並通過最後的 linear layer
        # (B, H, L_q, D_k) -> (B, L_q, H, D_k) -> (B, L_q, D)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class TransformerEncoderLayer(nn.Module):
    """
    單層 Encoder Layer。
    包含: Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        # 1. Multi-Head Attention (Self-Attention)
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        # 2. Add & Norm (Residual Connection)
        src = self.norm1(src + self.dropout1(attn_output))
        
        # 3. Feed Forward
        ff_output = self.feed_forward(src)
        # 4. Add & Norm
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    單層 Encoder Layer。
    包含: Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    """
    """
    單層 Decoder Layer。
    包含: Masked MHA -> Add & Norm -> MHA (Cross-Attention) -> Add & Norm -> FF -> Add & Norm
    禁止使用 nn.TransformerDecoderLayer
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor):
        # 1. Masked Multi-Head Attention (Self-Attention)
        attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        # 2. Add & Norm
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        
        # 3. Multi-Head Attention (Cross-Attention)
        # Q from decoder (tgt), K and V from encoder (memory)
        attn_output = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        # 4. Add & Norm
        tgt = self.norm2(tgt + self.dropout2(attn_output))
        
        # 5. Feed Forward
        ff_output = self.feed_forward(tgt)
        # 6. Add & Norm
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        
        return tgt


class Transformer(nn.Module): 
    """
    組合 Encoder 和 Decoder 的完整 Transformer。
    """
    def __init__(self, d_model: int, num_heads: int, num_encoder_layers: int, 
                 num_decoder_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """ 
        Encoder 堆疊。 
        Input src: (B, L_src, D)
        Input src_mask: (B, 1, 1, L_src)
        """
        x = src
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.norm_enc(x) # (B, L_src, D)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor):
        """ 
        Decoder 堆疊。 
        Input tgt: (B, L_tgt, D)
        Input memory: (B, L_src, D)
        Input tgt_mask: (B, 1, L_tgt, L_tgt)
        Input memory_mask: (B, 1, 1, L_src)
        """
        x = tgt
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm_dec(x) # (B, L_tgt, D)

    def forward(self, src_emb: torch.Tensor, tgt_emb: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor):
        """
        Transformer 的完整 forward pass (主要用於訓練)
        """
        # Encoder -> Memory
        memory = self.encode(src_emb, src_mask)
        # Decoder -> Output
        output = self.decode(tgt_emb, memory, tgt_mask, memory_mask)
        return output


class PositionalEncoding(nn.Module):
    """
    注入 token 在序列中位置的資訊。
    使用 sin 和 cos 函數的組合來生成位置編碼。
    """
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        pe = torch.zeros(maxlen, emb_size)
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        pe = pe.unsqueeze(0) # (1, maxlen, emb_size)

        # register_buffer 將 pe 存為模型的 state，但不會當作 parameter 來更新
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: torch.Tensor):
        # token_embedding shape: (batch_size, seq_len, emb_size)
        # self.pe[:, :token_embedding.size(1), :] 會取出 (1, seq_len, emb_size) 的位置編碼
        x = token_embedding + self.pe[:, :token_embedding.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    將輸入的 token 索引轉換為 embedding 向量。
    根據 "Attention Is All You Need" 論文，embedding 的權重會乘以 sqrt(d_model)。
    """
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        # tokens shape: (batch_size, seq_len)
        # output shape: (batch_size, seq_len, emb_size)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network (Top-level module)
# Hint: the masks should be carefully applied
class Seq2SeqNetwork(nn.Module):
    def __init__(self, 
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward,
                 dropout=0.1, 
                 device='cpu'):
        super().__init__()
        self.device=device
        self.transformer = Transformer(
            d_model=emb_size,
            num_heads=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def _create_padding_mask(self, tokens: torch.Tensor):
        # (B, L) -> (B, 1, 1, L)
        # PAD_IDX 來自 utils import
        return (tokens != PAD_IDX).unsqueeze(1).unsqueeze(1).to(self.device)

    def _create_causal_mask(self, size: int):
        # (L, L) -> (1, 1, L, L)
        # torch.tril 建立一個下三角矩陣 (diagonal=0)
        mask = torch.tril(torch.ones(size, size), diagonal=0).type(torch.bool)
        return mask.unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        訓練時的 forward pass。
        *** 加入了 mask ***
        """
        # src shape: (B, L_src)
        # tgt shape: (B, L_tgt)
        
        # 1. 建立 Masks
        # Encoder Self-Attention Mask (Padding Mask)
        src_mask = self._create_padding_mask(src) # (B, 1, 1, L_src)
        
        # Decoder Self-Attention Mask (Causal + Padding)
        tgt_padding_mask = self._create_padding_mask(tgt) # (B, 1, 1, L_tgt)
        tgt_causal_mask = self._create_causal_mask(tgt.size(1)) # (1, 1, L_tgt, L_tgt)
        tgt_mask = tgt_padding_mask & tgt_causal_mask # (B, 1, L_tgt, L_tgt)

        # Decoder Cross-Attention Mask (Encoder Padding Mask)
        memory_mask = src_mask # (B, 1, 1, L_src)

        # 2. Embedding + Positional Encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # 3. Transformer (修正後的呼叫)
        # 3a. 先用 Encoder 產生 memory
        memory = self.transformer.encode(src_emb, src_mask)
        # 3b. 再用 Decoder 產生輸出
        outs = self.transformer.decode(tgt_emb, memory, tgt_mask, memory_mask)
        # 4. Generator
        return self.generator(outs)


### ==================================================
### (2) TO-DO: Inference Function
### ==================================================
### Finish the translate function 
### Input: Chinese string
### Output: English string
def translate(model: torch.nn.Module, src_sentence: str, input_tokenizer, output_tokenizer, beam_width: int = 3, alpha: float = 0.7):
    """
    使用 Beam Search 進行翻譯
    :param model: 訓練好的模型
    :param src_sentence: 來源中文字串
    :param input_tokenizer: 中文 tokenizer
    :param output_tokenizer: 英文 tokenizer
    :param beam_width: 束搜索的寬度 (k)
    :param alpha: 長度懲罰 (length penalty) 的超參數
    """
    model.eval()
    
    # 0. 取得特殊 token 的 ID 和 device
    # 這些都來自 from utils import *
    BOS_IDX, EOS_IDX, PAD_IDX = 101, 102, 0 
    DEVICE = model.device # 從模型取得 device
    
    # 1. 將來源句子 tokenize 並轉換為 tensor
    src_tokens = input_tokenizer.encode(src_sentence)
    src_tensor = torch.tensor(src_tokens).view(1, -1).to(DEVICE) # (1, L_src)
    
    # 2. 建立來源句子的 padding mask
    src_mask = (src_tensor != PAD_IDX).unsqueeze(1).unsqueeze(1).to(DEVICE) # (1, 1, 1, L_src)
    
    # 3. Encoder 只需計算一次
    src_emb = model.positional_encoding(model.src_tok_emb(src_tensor))
    memory = model.transformer.encode(src_emb, src_mask) # (1, L_src, D)
    
    # 4. Beam Search Decoding
    
    # 初始化：beams 是一個 list，儲存 (sequence, score)
    # sequence 是 (1, len) 的 tensor, score 是該序列的累計 log probability
    initial_seq = torch.ones(1, 1).fill_(BOS_IDX).long().to(DEVICE) # (1, 1)
    beams = [(initial_seq, 0.0)]
    finished_beams = []

    for _ in range(MAX_LEN - 1):
        new_candidates = []
        
        for seq, score in beams:
            # (a) 檢查是否已完成 (是否以 <EOS> 結尾)
            last_token = seq[0, -1].item()
            if last_token == EOS_IDX:
                # 這個序列已完成，加入 finished_beams 列表
                finished_beams.append((seq, score))
                continue
            
            # (b) 取得目前序列的 embedding
            tgt_emb = model.positional_encoding(model.tgt_tok_emb(seq)) # (1, len, D)
            
            # (c) 建立 target 的 causal mask
            tgt_causal_mask = model._create_causal_mask(seq.size(1)) # (1, 1, len, len)
            
            # (d) Decoder 計算
            # memory_mask (src_mask) 保持不變
            out = model.transformer.decode(tgt_emb, memory, tgt_causal_mask, src_mask)
            
            # (e) Generator 預測
            # 只需要看最後一個 token 的輸出
            prob = model.generator(out[:, -1]) # (1, V_tgt)
            
            # (f) 取得 log 機率
            log_probs = torch.log_softmax(prob, dim=-1).squeeze(0) # (V_tgt)
            
            # (g) 選擇機率最高的 k 個 (k=beam_width)
            top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width) # (k)
            
            # (h) 將這 k 個候選人加入 new_candidates
            for i in range(beam_width):
                next_token_id = top_k_indices[i].item()
                log_prob = top_k_log_probs[i].item()
                
                new_seq = torch.cat([seq, torch.tensor([[next_token_id]]).to(DEVICE)], dim=1)
                new_score = score + log_prob
                
                new_candidates.append((new_seq, new_score))

        # (i) 如果沒有新的候選 (所有 beam 都完成了)，就提早結束
        if not new_candidates:
            break
            
        # (j) 排序所有候選人 (k * k 個)，並選出分數最高的 k 個
        ordered = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
        
        # (k) 如果新的 top-k 都已完成，也提早結束
        if all(b[0][0, -1].item() == EOS_IDX for b in beams):
            finished_beams.extend(beams)
            break

    # 5. 處理所有完成的 (或未完成的) 句子
    # 將最後還在 beams 裡的句子 (可能還沒 <EOS>) 也加入候選
    finished_beams.extend(beams)
    
    # 6. 使用長度懲罰 (Length Penalty) 找出最佳句子
    # alpha=0.7 是常用的值
    # 我們要找 score / (length^alpha) 最高的
    best_seq = None
    best_score = -float('inf')

    for seq, score in finished_beams:
        length = seq.size(1)
        # 避免除以零 (雖然 BOS 至少長度為 1)
        length_penalty = length ** alpha if length > 0 else 1.0
        normalized_score = score / length_penalty
        
        if normalized_score > best_score:
            best_score = normalized_score
            best_seq = seq
            
    # 7. 將 token ID 序列解碼回句子
    if best_seq is None:
        return "" # 預防萬一
        
    output_tokens = best_seq.squeeze(0).tolist()
    output_sentence = output_tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    return output_sentence

### ==================================================
### (3) TO-DO: load model
### ==================================================
### You can modify the hyper parameter below
def load_model(MODEL_PATH=None): 
    # --- 參數修改 ---
    # 原始參數 (EMB_SIZE=1024, NHEAD=64, FFN_HID_DIM=2048, LAYERS=6) 
    # 會導致模型 > 200M，遠超 100M 限制。
    #
    # 以下參數可將模型縮減至約 26.5M，符合實驗要求。
    EMB_SIZE = 256            # 論文中 D_model
    NHEAD = 8                 # 論文中 h (必須能整除 EMB_SIZE)
    FFN_HID_DIM = 1024        # 論文中 D_ff (4 * EMB_SIZE)
    NUM_ENCODER_LAYERS = 3    # 論文中 N
    NUM_DECODER_LAYERS = 3    # 論文中 N
    SRC_VOCAB_SIZE = tokenizer_chinese().vocab_size
    TGT_VOCAB_SIZE = tokenizer_english().vocab_size
    DROPOUT = 0.2
    model = Seq2SeqNetwork(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT, device=DEVICE)
    if MODEL_PATH is not None: 
        model.load_state_dict(torch.load(MODEL_PATH))
    return model
