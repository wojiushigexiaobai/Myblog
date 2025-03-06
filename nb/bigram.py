import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 64 # 每批处理的数据量
block_size = 256 # 上下文长度
# max_iters = 3000 # 最大迭代次数
max_iters = 5000
# eval_interval = 300 # 评估间隔
eval_interval = 500
# learning_rate = 1e-2 # 学习率
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备
eval_iters = 200 # 评估次数
n_embd = 384 # 嵌入维度
n_head = 6 # 多头注意力的数量
n_layer = 6 # 堆叠的Transformer块的数量
dropout = 0.2 # dropout的概率
# ---------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(r'E:\VScode_projects\a-python\somecode\PYTORCH_NN\dataset\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 这里是所有字符的集合
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 建立字符到索引的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 编码
decode = lambda l: ''.join([itos[i] for i in l]) # 解码

# 训练和测试数据
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 训练数据的长度
train_data = data[:n] # 训练数据
val_data = data[n:] # 验证数据

# 数据加载器
def get_batch(split):
    # 生成一个小批量的数据
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        # 计算注意力分数（“亲和性”）
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei,dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        # 执行值的加权聚合
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        # out = self.dropout(out)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    # 一个简单的线性层后面跟着一个非线性
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    # Transformer块：通信后跟计算
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we
        # want to parallelize across
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

"""
# self.sa_heads = MultiHeadAttention(4,n_embd//4) # 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd) # each position is handled by a single value vector
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd), # final layer norm
        # )

        # x = self.sa_heads(x) # apply one head of self-attenrion. (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
"""
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # 每个token直接从一个查找表中读取下一个token的logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # 每个token直接从一个查找表中读取下一个token的logits
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # idx是当前上下文的(B, T)数组的索引
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # 将idx裁剪为最后block_size个标记
            idx_cond = idx[:, -block_size:]
            # get the predictions
            # 获得预测
            logits, loss = self(idx_cond)
            # focus only on the last time step
            # 只关注最后一个时间步
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            # 应用softmax来获得概率
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            # 将采样的索引附加到运行序列
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
# 打印模型中的参数数量
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    # 每隔一段时间评估训练和验证集的损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    # 采样一批数据
    xb, yb = get_batch('train')

    # evaluate the loss
    # 评估损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
# 从模型中生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))