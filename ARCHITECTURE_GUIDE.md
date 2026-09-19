# Megatron-LM Architecture & Tensor Parallelism Guide

## Table of Contents
1. [Transformer Architecture](#transformer-architecture)
2. [Tensor Parallelism](#tensor-parallelism)
3. [Implementation Details](#implementation-details)
4. [Performance Considerations](#performance-considerations)

---

## Transformer Architecture

### Overview

The Megatron-LM style transformer uses **Pre-Layer Normalization (Pre-LN)** architecture, which differs from the original BERT/GPT Post-LN design:

```
Input
  ↓
LayerNorm → Attention → Residual Connection (+)
  ↓
LayerNorm → Feed-Forward → Residual Connection (+)
  ↓
Output
```

### Key Components

#### 1. Multi-Head Self-Attention

**Purpose**: Allow the model to attend to different positions with different representation subspaces

**Mathematical Formulation**:

```
Q = X * W_Q
K = X * W_K  
V = X * W_V

Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V

MultiHead(X) = Concat(head_1, ..., head_h) * W_O
```

Where:
- `h` = number of attention heads (e.g., 8, 12, 16)
- `d_k` = dimension per head = hidden_size / num_heads
- `√d_k` = scaling factor to prevent gradients from being too small

**Implementation Details**:
```python
# Q, K, V projections: hidden_size → hidden_size
query = nn.Linear(hidden_size, hidden_size)
key = nn.Linear(hidden_size, hidden_size)
value = nn.Linear(hidden_size, hidden_size)

# Reshape for multi-head: (batch, seq, hidden) → (batch, seq, num_heads, head_dim)
# Transpose to: (batch, num_heads, seq, head_dim)

# Attention scores: (batch, num_heads, seq, seq)
scores = Q @ K^T / √d_k

# Apply softmax and multiply by V
attention_weights = softmax(scores)
output = attention_weights @ V

# Concatenate heads and project: (batch, seq, hidden)
output = Linear(hidden_size, hidden_size)(concat_heads)
```

**Why Pre-LN?**
- More stable training (no "warmup" period needed)
- Better gradient flow
- Easier to scale to very deep models
- Avoids residual connection vanishing gradient problem

#### 2. Feed-Forward Network (FFN)

**Purpose**: Add non-linearity and increase model capacity

**Architecture**: Two dense layers with activation in between

```
Dense(hidden_size → intermediate_size) 
  → Activation (GELU/ReLU/SwiGLU)
  → Dense(intermediate_size → hidden_size)
```

**Typical expansion ratio**: intermediate_size = 4 * hidden_size

**Activation Functions**:

1. **GELU (Gaussian Error Linear Unit)**
   - Smooth approximation of ReLU
   - Better gradient flow
   - Default choice in modern transformers

2. **ReLU (Rectified Linear Unit)**
   - Simpler, faster
   - Can suffer from dead ReLUs
   - Older choice

3. **SwiGLU (Swish + GLU)**
   - SwiGLU(x, W) = (Swish(xW_1) ⊗ (xW_2))
   - Better performance but 1.33x more parameters
   - Used in PaLM, Llama models

#### 3. Layer Normalization

```
LayerNorm(x) = (x - mean) / sqrt(var + eps) * γ + β
```

Where:
- Normalization happens along feature dimension (hidden_size)
- Per-sample statistics (not batch statistics)
- Learnable scale (γ) and shift (β) parameters
- Small epsilon for numerical stability

**Why Pre-LN?**
- LayerNorm operates on same-scale input
- Prevents activation magnitude explosion
- Improves gradient stability

---

## Tensor Parallelism

### Concept

Tensor Parallelism splits tensors (weights and activations) across multiple GPUs to:
1. Reduce memory per GPU
2. Enable training of larger models
3. Increase throughput

### Types of Tensor Parallelism

#### 1. Column-Parallel Linear Layer (Output Parallelism)

**Architecture**:
```
Weight matrix W: [input_features, output_features]
Split along output dimension (columns):
  GPU 0: W[:, 0:out_features/2]
  GPU 1: W[:, out_features/2:out_features]
```

**Forward Pass**:
```
Input X: [batch, seq_len, input_features]
GPU 0: Y_0 = X @ W_0          → [batch, seq_len, out_features/2]
GPU 1: Y_1 = X @ W_1          → [batch, seq_len, out_features/2]
Output: Concat([Y_0, Y_1])    → [batch, seq_len, output_features]
```

**Communication**: No all-reduce needed! Each GPU just computes its portion.

**Example in MHA**:
- QKV projections use column parallelism
- Each GPU computes its portion of attention heads

#### 2. Row-Parallel Linear Layer (Input Parallelism)

**Architecture**:
```
Weight matrix W: [input_features, output_features]
Split along input dimension (rows):
  GPU 0: W[0:input_features/2, :]
  GPU 1: W[input_features/2:input_features, :]
```

**Forward Pass**:
```
Input X: [batch, seq_len, input_features]
Split X along feature dimension:
  GPU 0: X[:, :, 0:in_features/2]
  GPU 1: X[:, :, in_features/2:in_features]

Local computation:
  GPU 0: Y_0 = X_0 @ W_0      → [batch, seq_len, output_features]
  GPU 1: Y_1 = X_1 @ W_1      → [batch, seq_len, output_features]

All-reduce to sum contributions:
  Y = Y_0 + Y_1               → [batch, seq_len, output_features]
```

**Communication**: All-reduce needed to sum partial results!

**Example in FFN**:
- Output projection uses row parallelism
- Results summed across GPUs

### Transformer Block with TP

```
Transformer Block with Tensor Parallelism:

Input X
  ↓
LayerNorm
  ↓
Column-Parallel (Q, K, V projections)    ← Different heads on different GPUs
  ↓
Attention (local computation)
  ↓
Output projection                         ← Reduces and combines
  ↓
Residual connection
  ↓
LayerNorm
  ↓
Column-Parallel (Dense 1)                ← Expand to intermediate size
  ↓
Activation
  ↓
Row-Parallel (Dense 2)                   ← All-reduce to sum
  ↓
Residual connection
  ↓
Output Y
```

### Communication Pattern

```
Attention heads:
  GPU 0: Heads 0-3
  GPU 1: Heads 4-7
  GPU 2: Heads 8-11
  GPU 3: Heads 12-15

Forward:
  1. Column-parallel: Each GPU computes its heads (NO COMM)
  2. Attention: Each GPU computes attention locally
  3. Output projection: All-gather OR concat (minimal)

MLP:
  1. Column-parallel: Expand to intermediate (NO COMM)
  2. Row-parallel: Contract to hidden size (ALL-REDUCE)
```

### Communication vs Computation Trade-off

**All-reduce operations** are expensive. Cost analysis:

```
Bandwidth (GB/s):
  GPU Memory: ~900 GB/s (NVLink)
  Network:    ~12-100 GB/s (depends on infrastructure)

Communication Time (for 7B model, 4 GPUs):
  All-reduce time:     ~10-20 ms per step
  Computation time:    ~50 ms per step
  
Overhead: 10-20% of total training time
```

**Optimization**: Use high-bandwidth interconnects (NVLink, InfiniBand)

---

## Implementation Details

### Module Organization

```python
# Core modules
├── transformer_block.py
│   ├── MultiHeadAttention
│   ├── FeedForwardNetwork
│   ├── TransformerBlock
│   └── MegatronTransformer
│
└── tensor_parallel.py
    ├── ColumnParallelLinear
    ├── RowParallelLinear
    └── VocabParallelEmbedding
```

### Key Implementation Points

#### 1. Residual Connections

**Pre-LN residual**:
```python
# Attention block
x_norm = LayerNorm(x)
attn_out = Attention(x_norm)
x = x + attn_out  # Residual

# FFN block
x_norm = LayerNorm(x)
mlp_out = MLP(x_norm)
x = x + mlp_out   # Residual
```

**Why this matters**:
- Gradient flows through skip connections
- Prevents vanishing gradients
- Allows training of very deep models (100+ layers)

#### 2. Attention Mask

**Causal Mask** (for autoregressive generation):
```
Position:   0  1  2  3
      0   [✓  ✗  ✗  ✗]
      1   [✓  ✓  ✗  ✗]
      2   [✓  ✓  ✓  ✗]
      3   [✓  ✓  ✓  ✓]

Implementation:
mask = tril(ones(seq_len, seq_len))
attn_mask = (1 - mask) * -10000.0
scores = scores + attn_mask
```

**Padding Mask** (for variable-length sequences):
```python
# attention_mask: 1 for real tokens, 0 for padding
attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
```

#### 3. Gradient Clipping

**Why it's needed**:
- Loss can spike, causing exploding gradients
- Common in RNNs and transformers

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Typical values**: 1.0 for transformers

#### 4. Learning Rate Scheduling

**Common schedules**:
1. **Linear warmup + Cosine decay**
2. **Constant learning rate** (with warmup)
3. **Step decay** (reduce by 0.1 every N steps)

**Example**:
```python
def get_lr(step, total_steps, base_lr=1e-3):
    warmup_steps = 1000
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + cos(π * progress))
```

---

## Performance Considerations

### Memory Usage

**Model parameters**:
```
For 7B model:
  - Float32: 7B × 4 bytes = 28 GB
  - Float16: 7B × 2 bytes = 14 GB
  - With optimizer (Adam): 3× parameters = 84 GB (FP32)
```

**Activation memory** (during training):
```
Roughly: batch_size × seq_length × hidden_size × num_layers × 3
         (3× for q, k, v or 4× with gradients)

Example: batch=32, seq=2048, hidden=4096, layers=32
= 32 × 2048 × 4096 × 32 × 4 bytes
≈ 128 GB
```

**Tensor parallelism benefit**:
```
Without TP (1 GPU):
  - Model: 28 GB
  - Activations: 128 GB
  - Total: ~156 GB (won't fit!)

With TP=4 (4 GPUs):
  - Model per GPU: 7 GB (4 copies needed, ~28 GB total)
  - Activations per GPU: 32 GB
  - Total: ~39 GB per GPU (fits on 40GB GPU!)
```

### Throughput

**Factors affecting throughput**:
1. **Batch size**: Larger batches → higher utilization
2. **Sequence length**: Longer sequences → more computation
3. **Hardware**: GPU memory bandwidth, interconnect speed
4. **Communication overhead**: Tensor parallelism communication

**Typical throughput** (tokens/GPU/second):
```
V100 GPU, hidden=768, seq=512, batch=16:
  - Without TP: ~5,000 tokens/s
  - With TP=2: ~4,500 tokens/s (10% overhead)
  - With TP=4: ~3,500 tokens/s (30% overhead)
```

**Scaling efficiency**: Communication overhead increases with number of GPUs

### Recommended TP Sizes

```
Model size    Recommended TP   Reason
─────────────────────────────────────────────
< 7B          TP=1,2          Minimal communication needed
7B - 13B      TP=2,4          Balanced compute/comm
13B - 70B     TP=4,8          Significant memory savings
> 70B         TP=8,16         Must use TP to fit in memory
```

### Mixed Precision Training

**Using FP16 or BF16** (instead of FP32):
- 2× speedup
- 2× memory savings
- Need loss scaling to prevent underflow

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = loss_fn(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Multi-GPU Scaling

**Data Parallelism** (standard):
```
GPU 0: Model + Batch 0
GPU 1: Model + Batch 1
GPU 2: Model + Batch 2
GPU 3: Model + Batch 3

Gradient averaging across GPUs (all-reduce)
```

**Tensor Parallelism** (for large models):
```
GPU 0,1,2,3: Shared model (different parts)
             Process same batch together
             Communication between GPUs during forward/backward
```

**Pipeline Parallelism** (for very large models):
```
GPU 0: Layers 0-4
GPU 1: Layers 5-8
GPU 2: Layers 9-12
GPU 3: Layers 13-16

Process batches in stages (more complex)
```

### Best Practices

1. **Use AMP (Automatic Mixed Precision)**
   ```python
   from torch.amp import autocast
   with autocast(device_type="cuda"):
       # training code
   ```

2. **Gradient checkpointing** (save memory):
   ```python
   from torch.utils.checkpoint import checkpoint
   hidden = checkpoint(transformer_block, hidden, attention_mask)
   ```

3. **Gradient accumulation** (larger effective batch):
   ```python
   for micro_batch in batches:
       loss = model(micro_batch) / accumulation_steps
       loss.backward()
   optimizer.step()
   ```

4. **Learning rate warmup**:
   ```python
   # First 10,000 steps: linear warmup
   # Then: cosine decay
   ```

5. **Distributed training**:
   ```python
   model = DDP(model, device_ids=[rank])
   # Automatic gradient synchronization
   ```

---

## References

- **Megatron-LM Paper**: https://arxiv.org/abs/1909.08053
- **Tensor Parallelism Paper**: https://arxiv.org/abs/2104.04473
- **LLaMA Paper**: https://arxiv.org/abs/2302.13971
- **FlashAttention**: https://arxiv.org/abs/2205.14135
- **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html

---

## FAQ

**Q: When should I use tensor parallelism?**
A: When model weights don't fit in a single GPU. For models > 7B parameters, TP usually helps.

**Q: What's the communication overhead?**
A: ~10-30% depending on hardware. All-reduce is the bottleneck.

**Q: Can I combine TP with Data Parallelism?**
A: Yes! This is common: DP × TP = total GPUs. Usually (DP=2, TP=4) for 8 GPUs.

**Q: What about pipeline parallelism?**
A: More complex but can be combined with TP and DP for training 100B+ models.

**Q: How do I debug TP issues?**
A: Check rank/world_size, ensure divisibility, verify allreduce ops.
