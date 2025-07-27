import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, context_len, dropout_rate=0.1):
        super().__init__()

        # Define key, query, and value linear layers
        self.k = nn.Linear(embed_dim, head_dim, bias=False)
        self.q = nn.Linear(embed_dim, head_dim, bias=False)
        self.v = nn.Linear(embed_dim, head_dim, bias=False)

        self.head_dim = head_dim

        # Create a mask for attention
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        _, context, _ = x.shape  # dim (batch_size, context_len, embed_dim)

        # Compute keys, queries and values
        keys = self.k(x)  # dim (batch_size, context_len, head_dim)
        queries = self.q(x)
        values = self.v(x)

        # Switch the last two dimension of keys to make matrix multiplication possible
        # Matmul: (batch_size, context, head_dim) @ (batch_size, head_dim, context) = (batch_size, context, context)
        weights = queries @ keys.transpose(-2, -1) * (self.head_dim)**(-0.5)  # Scaling

        # Slice only the size that is needed from lower triangular matrix
        weights = weights.masked_fill(self.tril[:context, :context] == 0, float('-inf'))

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        # Matmul: (batch_size, context, context) @ (batch_size, context, head_dim) = (batch_size, contxt, head_dim)
        out = weights @ values

        return out
    
    
class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, context_len, dropout_rate=0.1):
        super().__init__()

        # Standard way to compute head_dim
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        head_dim = embed_dim // n_heads

        # Initiate multiple SingleHeadAttention layers to form multihead attention
        self.attention_heads = nn.ModuleList([SingleHeadAttention(embed_dim, head_dim, context_len, dropout_rate) for _ in range(n_heads)])

        # Attention projection linear layer
        self.attention_projection  = nn.Linear(n_heads * head_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        # Standard feedforward network in the Transformer block
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), # Standard to expand 4x
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ffn_dropout = nn.Dropout(0.1)

        # Layer mormalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        norm_x = self.ln1(x)
        # Compute the outputs from single head attention
        head_outputs = [head(norm_x) for head in self.attention_heads]  # dim (batch_size, context, head_dim, n_heads)
        # Concat the output from multiple attention blocks along n_heads dimension
        multihead_att_out = torch.cat(head_outputs, dim=-1)  # dim (batch_size, context, n_heads*head_dim)

        attention_out = self.attention_projection(multihead_att_out)  # dim (batch_size, context, embed_dim)
        x = x + self.attention_dropout(attention_out)  # Residual connection

        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)  # dim (batch_size, context, embed_dim)
        x = x + self.ffn_dropout(ffn_out)  # Residual connection

        return x
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, n_heads, n_blocks, dropout_rate=0.1):
        super().__init__()

        # Embedding layers for the tokens and positions
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_len, embed_dim)  # Using learned positional embeddings
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Initialize the Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(n_heads, embed_dim, context_len, dropout_rate) for _ in range(n_blocks)])

        self.ln_final = nn.LayerNorm(embed_dim)

        self.language_linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        _, context_length = idx.shape  # dim (batch_size, context_len)

        # Embed the tokens
        token_emb = self.token_embedding(idx)  # dim (batch_size, context_len, embed_dim)

        # Embed the positions
        pos = torch.arange(0, context_length, device=idx.device)  # dim (context_len)
        pos_embed = self.position_embedding(pos)  # dim (context_len, embed_dim)

        # Sum the embeddings, broadcasting happening because of different dimensions
        x = pos_embed + token_emb
        x = self.embedding_dropout(x)

        # Pass the embedded input to the Transformer blocks
        for block in self.blocks:
            x = block(x)  # dim (batch_size, context, embed_dim)

        x = self.ln_final(x)

        logits = self.language_linear(x)  # dim (batch_size, context, vocab_size)

        return logits