"""
Example usage and training scripts for Megatron-LM Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer_block import MegatronTransformer


def example_1_simple_inference():
    """Example 1: Simple inference with a pretrained model"""
    print("=" * 50)
    print("Example 1: Simple Inference")
    print("=" * 50)
    
    # Create model
    model = MegatronTransformer(
        vocab_size=10000,
        hidden_size=256,
        num_blocks=6,
        num_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
    )
    
    # Set to eval mode
    model.eval()
    
    # Create sample input
    batch_size, seq_length = 2, 20
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :10].tolist()}")
    print()


def example_2_training():
    """Example 2: Training on synthetic data"""
    print("=" * 50)
    print("Example 2: Training on Synthetic Data")
    print("=" * 50)
    
    # Hyperparameters
    vocab_size = 1000
    hidden_size = 128
    batch_size = 8
    seq_length = 32
    num_epochs = 3
    learning_rate = 1e-4
    
    # Create model
    model = MegatronTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_blocks=4,
        num_heads=4,
        intermediate_size=512,
        max_position_embeddings=256,
    )
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create synthetic dataset
    num_samples = 100
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    total_loss = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
            # Forward pass
            logits = model(batch_input_ids)
            loss = loss_fn(logits.view(-1, vocab_size), batch_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        total_loss += avg_loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print(f"Average loss: {total_loss / num_epochs:.4f}")
    print()


def example_3_inference_with_attention_mask():
    """Example 3: Inference with attention mask"""
    print("=" * 50)
    print("Example 3: Inference with Attention Mask")
    print("=" * 50)
    
    model = MegatronTransformer(
        vocab_size=5000,
        hidden_size=256,
        num_blocks=4,
        num_heads=8,
        intermediate_size=1024,
    )
    model.eval()
    
    # Create input with variable sequence lengths
    batch_size = 2
    max_seq_length = 30
    
    input_ids = torch.randint(0, 5000, (batch_size, max_seq_length))
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones(batch_size, max_seq_length)
    attention_mask[0, 20:] = 0  # Pad first sample
    attention_mask[1, 25:] = 0  # Pad second sample
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Attention mask (first sample): {attention_mask[0].tolist()}")
    print()


def example_4_model_comparison():
    """Example 4: Compare different model sizes"""
    print("=" * 50)
    print("Example 4: Model Size Comparison")
    print("=" * 50)
    
    configs = [
        {
            "name": "Small",
            "hidden_size": 128,
            "num_blocks": 4,
            "num_heads": 4,
            "intermediate_size": 512,
        },
        {
            "name": "Medium",
            "hidden_size": 256,
            "num_blocks": 8,
            "num_heads": 8,
            "intermediate_size": 1024,
        },
        {
            "name": "Large",
            "hidden_size": 512,
            "num_blocks": 12,
            "num_heads": 16,
            "intermediate_size": 2048,
        },
    ]
    
    vocab_size = 10000
    
    print(f"{'Model':<15} {'Params':<15} {'Memory (MB)':<15}")
    print("-" * 45)
    
    for config in configs:
        model = MegatronTransformer(
            vocab_size=vocab_size,
            hidden_size=config["hidden_size"],
            num_blocks=config["num_blocks"],
            num_heads=config["num_heads"],
            intermediate_size=config["intermediate_size"],
        )
        
        num_params = model.get_num_parameters()
        memory_mb = num_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        print(f"{config['name']:<15} {num_params:<15,} {memory_mb:<15.2f}")
    
    print()


def example_5_gradient_accumulation():
    """Example 5: Training with gradient accumulation"""
    print("=" * 50)
    print("Example 5: Training with Gradient Accumulation")
    print("=" * 50)
    
    # Hyperparameters
    vocab_size = 1000
    hidden_size = 128
    batch_size = 4
    accumulation_steps = 4
    seq_length = 32
    num_epochs = 2
    
    model = MegatronTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_blocks=4,
        num_heads=4,
        intermediate_size=512,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create synthetic dataset
    num_samples = 100
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    
    accumulated_loss = 0
    step = 0
    
    for epoch in range(num_epochs):
        for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
            # Forward pass
            logits = model(batch_input_ids)
            loss = loss_fn(logits.view(-1, vocab_size), batch_labels.view(-1))
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            accumulated_loss += loss.item()
            step += 1
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Step {step // accumulation_steps}, Accumulated Loss: {accumulated_loss:.4f}")
                accumulated_loss = 0
    
    print()


def example_6_token_generation():
    """Example 6: Simple token generation (greedy decoding)"""
    print("=" * 50)
    print("Example 6: Token Generation (Greedy Decoding)")
    print("=" * 50)
    
    model = MegatronTransformer(
        vocab_size=1000,
        hidden_size=256,
        num_blocks=6,
        num_heads=8,
        intermediate_size=1024,
        max_position_embeddings=256,
    )
    model.eval()
    
    # Start token (could be special token)
    start_token = torch.tensor([[1]])  # batch_size=1, seq_length=1
    
    generated_tokens = [1]
    
    with torch.no_grad():
        current_tokens = start_token
        
        # Generate 20 more tokens
        for _ in range(20):
            # Get predictions
            logits = model(current_tokens)
            
            # Get the last token prediction
            next_token_logits = logits[:, -1, :]
            
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated_tokens.append(next_token.item())
            
            # Prepare for next iteration (keep all tokens for context)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
    
    print(f"Start token: {start_token[0, 0].item()}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generated sequence length: {len(generated_tokens)}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 50)
    print("MEGATRON-LM EXAMPLES")
    print("=" * 50 + "\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    example_1_simple_inference()
    example_2_training()
    example_3_inference_with_attention_mask()
    example_4_model_comparison()
    example_5_gradient_accumulation()
    example_6_token_generation()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
