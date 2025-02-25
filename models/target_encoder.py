import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TargetEncoder(nn.Module):
    """
    Target Encoder for Text-JEPA
    
    This encoder processes the target tokens and produces representations.
    In the diagram, this is the "Target Encoder (12-layer Transformer with RoPE)".
    
    The architecture is identical to the Context Encoder, but it's updated
    via exponential moving average (EMA) of the context encoder parameters.
    """
    
    def __init__(
        self,
        model_name_or_path="roberta-base",
        hidden_size=768,
        num_layers=12,
        use_custom_model=False,
        dropout_prob=0.1,
    ):
        """
        Initialize the Target Encoder.
        
        Args:
            model_name_or_path: Pretrained model name or path
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            use_custom_model: If True, initialize a custom transformer model instead of using pretrained
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        if use_custom_model:
            # Initialize a custom transformer model
            config = AutoConfig.from_pretrained(model_name_or_path)
            config.num_hidden_layers = num_layers
            config.hidden_size = hidden_size
            config.hidden_dropout_prob = dropout_prob
            config.attention_probs_dropout_prob = dropout_prob
            
            self.encoder = AutoModel.from_config(config)
        else:
            # Load pretrained model
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            
        # Note: This encoder is identical to the Context Encoder
        # Its parameters will be initialized and updated with EMA in the main Text-JEPA model
            
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the Target Encoder.
        
        Args:
            input_ids: Tensor of token ids [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            outputs: Encoded representations [batch_size, seq_length, hidden_size]
        """
        # If no attention mask is provided, create one (all 1s)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Encode the inputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        return hidden_states
