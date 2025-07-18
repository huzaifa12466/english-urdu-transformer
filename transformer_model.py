import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    """Returns the appropriate device (CUDA if available, else CPU)"""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scale_dot_product(q, k, v, mask=None):
    """Implements scaled dot-product attention"""
    d_k = q.size()[-1]
    # Scale the dot product by the square root of the key dimension
    scale = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)
        scale = scale + mask
    
    # Apply softmax to get attention weights
    attention = F.softmax(scale, dim=-1)
    # Compute weighted sum of values
    value = torch.matmul(attention, v)
    return value, attention

class PositionalEncoding(nn.Module):
    """Adds positional information to token embeddings"""
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self):
        # Create even indices for positional encoding
        even_i = torch.arange(0, self.d_model, 2).float()
        # Calculate denominator for positional encoding formula
        denominator = torch.pow(10000, even_i/self.d_model)
        # Create position tensor
        position = torch.arange(self.max_len).reshape(self.max_len, 1)
        # Calculate sine and cosine encodings
        even_pe = torch.sin(position/denominator)
        odd_pe = torch.cos(position/denominator)
        # Stack and flatten the encodings
        stacked = torch.stack([even_pe, odd_pe], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    """Converts sentences to embeddings with positional encoding"""
    def __init__(self, d_model, max_len, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super(SentenceEmbedding, self).__init__()
        self.vocab_size = len(language_to_index)
        self.max_len = max_len
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.language_to_index = language_to_index

    def batch_tokenize(self, batch, start_token=True, end_token=True):
        """Tokenizes a batch of sentences"""
        def tokenize(sentence):
            tokens = sentence
            sentence_indices = []

            # Convert tokens to indices
            for token in tokens:
                if token in self.language_to_index:
                    sentence_indices.append(self.language_to_index[token])
            
            # Add start token
            if start_token:
                sentence_indices.insert(0, self.language_to_index[self.START_TOKEN])
            # Add end token
            if end_token:
                sentence_indices.append(self.language_to_index[self.END_TOKEN])

            # Add padding
            while len(sentence_indices) < self.max_len:
                sentence_indices.append(self.language_to_index[self.PADDING_TOKEN])

            # Truncate if necessary
            if len(sentence_indices) > self.max_len:
                sentence_indices = sentence_indices[:self.max_len]

            return torch.tensor(sentence_indices, dtype=torch.long)

        tokenized = [tokenize(sentence) for sentence in batch]
        return torch.stack(tokenized).to(get_device())

    def forward(self, x, start_token=True, end_token=True):
        # Tokenize input and get embeddings
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        # Add positional encoding
        x = x + self.positional_encoding().to(get_device())
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    """Implements multi-head attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, seq_length, d_model = x.size()
        # Project input to Q, K, V
        qkv = self.qkv_layer(x)
        # Reshape for multi-head attention
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        # Apply scaled dot-product attention
        values, attention = scale_dot_product(q, k, v, mask)
        # Reshape back to original dimensions
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)
        # Final linear projection
        out = self.linear_layer(values)
        return out, attention

class LayerNormalization(nn.Module):
    """Implements layer normalization"""
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        # Calculate mean and variance
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        # Normalize and scale
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feed-forward network"""
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """Single encoder layer with multi-head attention and feed-forward network"""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        # Multi-head attention with residual connection
        residual_x = x
        x, _ = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        # Feed-forward network with residual connection
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    """Sequential stack of encoder layers"""
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    """Complete encoder with embedding and multiple encoder layers"""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(d_model, max_sequence_length, 
                                                  language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                        for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

class MultiheadCrossAttention(nn.Module):
    """Implements multi-head cross-attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(MultiheadCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, q_len, _ = x.size()
        _, kv_len, _ = y.size()
        # Project queries, keys, and values
        q = self.q_proj(x)
        kv = self.kv_proj(y)
        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.view(batch_size, kv_len, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)
        # Apply scaled dot-product attention
        attn_output, attn_weights = scale_dot_product(q, k, v, mask)
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.out_proj(attn_output)
        return output, attn_weights

class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and feed-forward network"""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.multihead_cross_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        # Self-attention with residual connection
        residual_y = y.clone()
        y, _ = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual_y)
        # Cross-attention with residual connection
        residual_y = y.clone()
        y, _ = self.multihead_cross_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + residual_y)
        # Feed-forward network with residual connection
        residual_y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_y)
        return y

class SequentialDecoder(nn.Sequential):
    """Sequential stack of decoder layers"""
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    """Complete decoder with embedding and multiple decoder layers"""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(d_model, max_sequence_length, 
                                                  language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                        for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks"""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, ur_vocab_size, english_to_index, urdu_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                             max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                              max_sequence_length, urdu_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, ur_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x, y, encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, decoder_cross_attention_mask=None,
                enc_start_token=False, enc_end_token=False,
                dec_start_token=False, dec_end_token=False):
        # Encode input sequence
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        # Decode with cross-attention to encoder output
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask,
                          start_token=dec_start_token, end_token=dec_end_token)
        # Final linear layer to vocabulary size
        out = self.linear(out)
        return out