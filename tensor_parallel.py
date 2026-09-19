"""
Tensor Parallelism Implementation for Megatron-LM
Includes Column-parallel and Row-parallel linear layers
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from torch.distributed import ProcessGroup


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism (output parallelism)
    
    The weight matrix is split along the output dimension (columns)
    Each rank computes a portion of the output features
    No communication needed in forward pass
    All-reduce needed in backward pass
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[ProcessGroup] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        
        if process_group is not None:
            world_size = dist.get_world_size(process_group)
        else:
            world_size = 1
        
        assert out_features % world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"
        
        self.out_features_per_rank = out_features // world_size
        
        # Each rank has a portion of the output features
        self.linear = nn.Linear(in_features, self.out_features_per_rank, bias=bias)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - each rank computes its portion of output features
        No synchronization needed
        
        Args:
            input_tensor: (batch_size, seq_length, in_features) or (batch_size, in_features)
        
        Returns:
            output: (batch_size, seq_length, out_features_per_rank) or (batch_size, out_features_per_rank)
        """
        return self.linear(input_tensor)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism (input parallelism)
    
    The weight matrix is split along the input dimension (rows)
    Each rank gets a portion of the input features
    All-reduce needed in forward pass to gather results
    No communication needed in backward pass
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[ProcessGroup] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        
        if process_group is not None:
            world_size = dist.get_world_size(process_group)
        else:
            world_size = 1
        
        assert in_features % world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"
        
        self.in_features_per_rank = in_features // world_size
        
        # Each rank has a portion of the input features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - each rank computes output for its portion of inputs,
        then all-reduce to sum contributions
        
        Args:
            input_tensor: (batch_size, seq_length, in_features_per_rank) or (batch_size, in_features_per_rank)
        
        Returns:
            output: (batch_size, seq_length, out_features) or (batch_size, out_features)
        """
        output = self.linear(input_tensor)
        
        # Sum contributions from all ranks
        if self.process_group is not None and dist.is_initialized():
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output


class TensorParallelLinear(nn.Module):
    """
    Wrapper for tensor-parallel linear layers
    Automatically chooses between column and row parallel based on config
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_type: str = "column",
        process_group: Optional[ProcessGroup] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        assert parallel_type in ["column", "row"], \
            f"parallel_type must be 'column' or 'row', got {parallel_type}"
        
        self.parallel_type = parallel_type
        
        if parallel_type == "column":
            self.linear = ColumnParallelLinear(
                in_features, out_features, process_group, bias
            )
        else:  # row
            self.linear = RowParallelLinear(
                in_features, out_features, process_group, bias
            )
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(input_tensor)


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism
    
    The vocabulary is split across ranks along the embedding dimension
    Each rank maintains a portion of vocabulary embeddings
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        process_group: Optional[ProcessGroup] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.process_group = process_group
        
        # Get distributed configuration
        if process_group is not None and dist.is_initialized():
            rank = dist.get_rank(process_group)
            world_size = dist.get_world_size(process_group)
        else:
            rank, world_size = 0, 1
        
        assert num_embeddings % world_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by world_size ({world_size})"
        
        self.vocab_size_per_rank = num_embeddings // world_size
        self.rank = rank
        self.world_size = world_size
        
        # Each rank has a portion of vocabulary
        self.embedding = nn.Embedding(
            self.vocab_size_per_rank,
            embedding_dim,
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices (batch_size, seq_length)
        
        Returns:
            embeddings: (batch_size, seq_length, embedding_dim)
        """
        # Adjust input IDs to local vocabulary range
        local_input_ids = input_ids - self.rank * self.vocab_size_per_rank
        
        # Mask out-of-range indices
        mask = (local_input_ids < 0) | (local_input_ids >= self.vocab_size_per_rank)
        local_input_ids = local_input_ids.clamp(0, self.vocab_size_per_rank - 1)
        
        # Get embeddings
        embeddings = self.embedding(local_input_ids)
        
        # Zero out embeddings for out-of-range tokens
        embeddings[mask] = 0
        
        return embeddings


class DistributedParallelConfig:
    """Configuration for distributed training with tensor parallelism"""
    
    def __init__(
        self,
        world_size: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        rank: int = 0,
        backend: str = "nccl",
    ):
        assert (world_size % (tensor_parallel_size * pipeline_parallel_size) == 0), \
            "world_size must be divisible by (tensor_parallel_size * pipeline_parallel_size)"
        
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.rank = rank
        self.backend = backend
        
        # Calculate derived quantities
        self.num_pipeline_stages = world_size // tensor_parallel_size
        self.num_tensor_parallel_groups = world_size // tensor_parallel_size
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")


def initialize_tensor_parallel(
    world_size: int,
    tensor_parallel_size: int,
    rank: int,
    backend: str = "nccl",
) -> tuple:
    """
    Initialize tensor parallelism groups
    
    Args:
        world_size: Total number of processes
        tensor_parallel_size: Number of processes per tensor parallel group
        rank: Rank of current process
        backend: Communication backend (nccl, gloo, etc.)
    
    Returns:
        (tensor_parallel_group, data_parallel_group)
    """
    
    assert world_size % tensor_parallel_size == 0
    
    # Create tensor parallel groups
    num_tp_groups = world_size // tensor_parallel_size
    tensor_parallel_groups = []
    
    for i in range(num_tp_groups):
        ranks = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
        group = dist.new_group(ranks, backend=backend)
        tensor_parallel_groups.append(group)
    
    # Create data parallel groups
    data_parallel_groups = []
    
    for i in range(tensor_parallel_size):
        ranks = list(range(i, world_size, tensor_parallel_size))
        group = dist.new_group(ranks, backend=backend)
        data_parallel_groups.append(group)
    
    # Get current process groups
    tp_group_idx = rank // tensor_parallel_size
    dp_group_idx = rank % tensor_parallel_size
    
    tp_group = tensor_parallel_groups[tp_group_idx]
    dp_group = data_parallel_groups[dp_group_idx]
    
    return tp_group, dp_group


def reduce_scatter_tensor_parallel(
    tensor: torch.Tensor,
    tensor_parallel_group: ProcessGroup,
    dim: int = -1,
) -> torch.Tensor:
    """
    Reduce-scatter operation for tensor parallelism
    
    Args:
        tensor: Input tensor to reduce-scatter
        tensor_parallel_group: Process group for communication
        dim: Dimension along which to scatter
    
    Returns:
        Reduced and scattered tensor
    """
    if tensor_parallel_group is None or not dist.is_initialized():
        return tensor
    
    # Perform reduce-scatter
    world_size = dist.get_world_size(tensor_parallel_group)
    
    if world_size == 1:
        return tensor
    
    # Split tensor along dimension
    split_size = tensor.shape[dim] // world_size
    chunks = torch.split(tensor, split_size, dim=dim)
    
    # Create output tensor of appropriate size
    output_shape = list(tensor.shape)
    output_shape[dim] = split_size
    output = torch.zeros(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Reduce-scatter
    input_list = list(chunks)
    dist.reduce_scatter(output, input_list, group=tensor_parallel_group)
    
    return output


def all_gather_tensor_parallel(
    tensor: torch.Tensor,
    tensor_parallel_group: ProcessGroup,
    dim: int = -1,
) -> torch.Tensor:
    """
    All-gather operation for tensor parallelism
    
    Args:
        tensor: Input tensor to gather
        tensor_parallel_group: Process group for communication
        dim: Dimension along which to gather
    
    Returns:
        Gathered tensor
    """
    if tensor_parallel_group is None or not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size(tensor_parallel_group)
    
    if world_size == 1:
        return tensor
    
    # Prepare for all-gather
    output_shape = list(tensor.shape)
    output_shape[dim] = output_shape[dim] * world_size
    output = torch.zeros(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Create output chunks for all_gather
    output_chunks = list(torch.split(output, tensor.shape[dim], dim=dim))
    
    # All-gather
    dist.all_gather_into_tensor(output, tensor, group=tensor_parallel_group)
    
    return output
