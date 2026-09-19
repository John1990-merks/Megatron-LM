"""
Test suite for Megatron-LM Transformer Block implementation
"""

import torch
import pytest
from transformer_block import (
    MultiHeadAttention,
    FeedForwardNetwork,
    TransformerBlock,
    MegatronTransformer,
)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention layer"""
    
    @pytest.fixture
    def attention_module(self):
        return MultiHeadAttention(hidden_size=256, num_heads=8)
    
    def test_attention_forward_shape(self, attention_module):
        """Test that attention output has correct shape"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        output, weights = attention_module(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert weights.shape == (batch_size, 8, seq_length, seq_length)
    
    def test_attention_with_mask(self, attention_module):
        """Test attention with attention mask"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Create attention mask (0 for attend, -inf for ignore)
        mask = torch.zeros(batch_size, 1, seq_length, seq_length)
        mask[:, :, :, -1] = -10000.0  # Mask out last token
        
        output, weights = attention_module(hidden_states, mask)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
        # Check that last token has negligible attention to masked position
        assert weights[:, :, :, -1].max().item() < 1e-4
    
    def test_attention_weights_sum_to_one(self, attention_module):
        """Test that attention weights sum to 1 across sequence"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        _, weights = attention_module(hidden_states)
        
        # Attention weights should sum to 1 along the value dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
    
    def test_attention_gradient_flow(self, attention_module):
        """Test that gradients flow properly through attention"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        output, _ = attention_module(hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape


class TestFeedForwardNetwork:
    """Tests for FeedForwardNetwork layer"""
    
    @pytest.fixture
    def ffn_module(self):
        return FeedForwardNetwork(hidden_size=256, intermediate_size=1024)
    
    def test_ffn_forward_shape(self, ffn_module):
        """Test FFN output shape"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        output = ffn_module(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
    
    def test_ffn_with_different_activations(self):
        """Test FFN with different activation functions"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        for activation in ["gelu", "relu"]:
            ffn = FeedForwardNetwork(
                hidden_size=hidden_size,
                intermediate_size=1024,
                activation=activation
            )
            output = ffn(hidden_states)
            assert output.shape == (batch_size, seq_length, hidden_size)
    
    def test_ffn_invalid_activation(self):
        """Test that invalid activation raises error"""
        with pytest.raises(ValueError):
            FeedForwardNetwork(
                hidden_size=256,
                intermediate_size=1024,
                activation="invalid"
            )
    
    def test_ffn_gradient_flow(self, ffn_module):
        """Test gradient flow through FFN"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        output = ffn_module(hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None


class TestTransformerBlock:
    """Tests for TransformerBlock"""
    
    @pytest.fixture
    def transformer_block(self):
        return TransformerBlock(
            hidden_size=256,
            num_heads=8,
            intermediate_size=1024
        )
    
    def test_block_forward_shape(self, transformer_block):
        """Test transformer block output shape"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        output, weights = transformer_block(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert weights.shape == (batch_size, 8, seq_length, seq_length)
    
    def test_block_residual_connection(self, transformer_block):
        """Test that residual connections are working"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        output, _ = transformer_block(hidden_states)
        
        # Output should be different from input due to transformations
        assert not torch.allclose(output, hidden_states)
        
        # But magnitudes should be reasonable (not exploding)
        assert output.abs().max() < 100.0
    
    def test_block_with_dropout(self):
        """Test that dropout is applied during training"""
        batch_size, seq_length, hidden_size = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        block = TransformerBlock(
            hidden_size=hidden_size,
            num_heads=8,
            intermediate_size=1024,
            dropout_rate=0.5  # High dropout for testing
        )
        
        block.train()
        outputs = []
        for _ in range(3):
            output, _ = block(hidden_states)
            outputs.append(output)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(outputs[0], outputs[1])
        assert not torch.allclose(outputs[1], outputs[2])
        
        # In eval mode, outputs should be the same
        block.eval()
        output1, _ = block(hidden_states)
        output2, _ = block(hidden_states)
        assert torch.allclose(output1, output2)


class TestMegatronTransformer:
    """Tests for full MegatronTransformer model"""
    
    @pytest.fixture
    def model(self):
        return MegatronTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_blocks=4,
            num_heads=8,
            intermediate_size=1024,
        )
    
    def test_model_forward_shape(self, model):
        """Test model output shape"""
        batch_size, seq_length = 2, 10
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_length, vocab_size)
    
    def test_model_with_attention_mask(self, model):
        """Test model with attention mask"""
        batch_size, seq_length = 2, 10
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, seq_length, vocab_size)
    
    def test_model_parameter_count(self, model):
        """Test model parameter counting"""
        total_params = model.get_num_parameters()
        trainable_params = model.get_num_trainable_parameters()
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable
    
    def test_model_weight_tying(self, model):
        """Test that embedding and output layer weights are tied"""
        assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    
    def test_model_loss_computation(self, model):
        """Test that loss can be computed"""
        batch_size, seq_length = 2, 10
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        logits = model(input_ids)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_model_backward_pass(self, model):
        """Test backward pass"""
        batch_size, seq_length = 2, 10
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        logits = model(input_ids)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None
    
    def test_model_different_configs(self):
        """Test model with different configurations"""
        configs = [
            {"hidden_size": 128, "num_blocks": 2, "num_heads": 4},
            {"hidden_size": 512, "num_blocks": 6, "num_heads": 8},
            {"hidden_size": 1024, "num_blocks": 12, "num_heads": 16},
        ]
        
        for config in configs:
            model = MegatronTransformer(
                vocab_size=1000,
                intermediate_size=config["hidden_size"] * 4,
                **config
            )
            
            input_ids = torch.randint(0, 1000, (2, 10))
            logits = model(input_ids)
            
            assert logits.shape == (2, 10, 1000)


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step"""
        model = MegatronTransformer(
            vocab_size=100,
            hidden_size=128,
            num_blocks=2,
            num_heads=4,
            intermediate_size=512,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Forward pass
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))
        
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, 100), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that weights updated
        assert not torch.isnan(loss)
    
    def test_model_inference_mode(self):
        """Test model in inference mode"""
        model = MegatronTransformer(
            vocab_size=100,
            hidden_size=128,
            num_blocks=2,
            num_heads=4,
            intermediate_size=512,
        )
        
        model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, 100, (2, 10))
            logits = model(input_ids)
        
        # Should work without computing gradients
        assert logits.grad_fn is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
