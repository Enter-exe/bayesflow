import keras
from keras import layers, ops, random
from bayesflow.transformers.lab import LinearAttention


class MultiHeadLinearAttention(layers.Layer):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            dropout=0.1, 
            use_bias=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_bias = use_bias

        # Create a LinearAttention module per head
        self.heads = [LinearAttention(self.head_dim, dropout_rate=dropout, use_bias=use_bias) for _ in range(num_heads)]

        # Final projection layer
        self.output_projection = layers.Dense(embed_dim)

    def split_heads(self, x):
        # Input shape: (batch_size, seq_len, embed_dim)
        x = ops.reshape(x, (ops.shape(x)[0], ops.shape(x)[1], self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, head_dim)

    def combine_heads(self, x):
        # Input shape: (batch_size, num_heads, seq_len, head_dim)
        x = ops.transpose(x, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        return ops.reshape(x, (ops.shape(x)[0], ops.shape(x)[1], self.embed_dim))  # (batch_size, seq_len, embed_dim)
    '''
    def call(self, query, key, value, training):
        # Split into heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Apply linear attention per head
        outputs = []
        for i in range(self.num_heads):
            out = self.heads[i](query[:, i], key[:, i], value[:, i], training=training)  # (batch, seq_len, head_dim)
            outputs.append(out)

        # Stack outputs: shape (batch_size, num_heads, seq_len, head_dim)
        stacked = ops.stack(outputs, axis=1)

        # Combine heads and project
        combined = self.combine_heads(stacked)
        return self.output_projection(combined)
    '''
    def call(self, inputs, training=False, return_attention=False):
        # Split into heads
        inputs = self.split_heads(inputs)

        # Apply linear attention per head
        outputs = []
        attn_weights = []
        for i in range(self.num_heads):
            context, attn = self.heads[i](inputs[:, i], training=training, return_attention=return_attention)  # (batch, seq_len, head_dim)
            outputs.append(context)
            if return_attention:
                attn_weights.append(attn)

        # Stack outputs: shape (batch_size, num_heads, seq_len, head_dim)
        stacked = ops.stack(outputs, axis=1)

        # Combine heads and project
        combined = self.combine_heads(stacked)
        out  = self.output_projection(combined)
        #if return_attention:
            #attn_stack = ops.stack(attn_weights, axis=2)
        return out#, attn_stack