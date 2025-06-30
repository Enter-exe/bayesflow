import keras
from keras import layers, ops, random
from bayesflow.networks.transformers.lab import LinearAttention


class MultiHeadLinearAttention(layers.Layer):
    """
    A Keras layer implementing multi-head linear attention.

    Splits input into multiple heads, applies linear attention independently,
    then combines and projects the output.
    """
    def __init__(
            self, 
            key_dim, 
            num_heads, 
            dropout=0.1, 
            use_bias=False,
            output_shape=None,
            **kwargs
    ):
        """
        Initializes the MultiHeadLinearAttention layer.

        Args:
            key_dim (int): Dimensionality for each attention head.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            use_bias (bool): Whether to use bias in attention heads.
            output_shape (int, optional): Final output dimension.
            **kwargs: Additional keyword arguments for base Layer.
        """    
        super().__init__(**kwargs)
        

        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.output_shape = (key_dim * num_heads if output_shape is None else output_shape)

        self.heads = [LinearAttention(feature_dim=key_dim, dropout_rate=dropout, use_bias=use_bias) for _ in range(num_heads)]

        self.output_projection = layers.Dense(self.output_shape, use_bias)

    def split_heads(self, x):
        """
        Splits input tensor into multiple attention heads.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            List[Tensor]: List of split tensors for each head.
        """
        x = ops.reshape(x, (ops.shape(x)[0], ops.shape(x)[1], self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))  

    def combine_heads(self, x):
        """
        Combines the output of all attention heads.

        Args:
            head_outputs (List[Tensor]): List of outputs from each attention head.

        Returns:
            Tensor: Combined output tensor.
        """
        x = ops.transpose(x, (0, 2, 1, 3)) 
        return ops.reshape(x, (ops.shape(x)[0], ops.shape(x)[1], self.embed_dim))  
    
    def call(self, query, value=None, key=None, training=False, return_attention=False):
        """
        Applies multi-head linear attention using the given query, key, and value tensors.

        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (Tensor): Key tensor of the same shape or compatible shape.
            value (Tensor): Value tensor of the same shape or compatible shape.
            training (bool): Whether the layer is in training mode (affects dropout).

        Returns:
            Tensor: Output tensor after applying attention and final projection.
        """
        if key is None:
            key = query
        if value is None:
            value = key

        
        q = self.split_heads(query)
        k = self.split_heads(key)
        v = self.split_heads(value)
        
        outputs = []
        attn_weights = []
        for i in range(self.num_heads):
            context, attn = self.heads[i](query=q[:, i], key=k[:, i], value=v[:, i], training=training, return_attention=return_attention) 
            outputs.append(context)
            if return_attention:
                attn_weights.append(attn)

        
        stacked = ops.stack(outputs, axis=1)

        combined = self.combine_heads(stacked)
        out  = self.output_projection(combined)
        #if return_attention:
            #attn_stack = ops.stack(attn_weights, axis=2)
        return out#, attn_stack