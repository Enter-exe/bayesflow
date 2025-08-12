import keras
from keras import layers, ops, random
from .linear_attention import LinearAttention

@serializable("bayesflow.networks")
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
        Initialize the MultiHeadLinearAttention layer.

        Parameters
        ----------
        key_dim : int
            Dimensionality for each attention head.
        num_heads : int
            Number of attention heads.
        dropout : float, optional
            Dropout rate (default is 0.1).
        use_bias : bool, optional
            Whether to use bias in attention heads (default is False).
        output_shape : int, optional
            Final output dimension. If None, uses key_dim * num_heads.
        **kwargs : dict
            Additional keyword arguments for the base Layer.
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
        Split the input tensor into multiple attention heads.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        List[Tensor]
            List of split tensors for each head.
        """

        x = ops.reshape(x, (ops.shape(x)[0], ops.shape(x)[1], self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))  

    def combine_heads(self, x):
        """
        Combine the outputs of all attention heads.

        Parameters
        ----------
        x : Tensor
            Tensor of shape (batch_size, num_heads, seq_len, head_dim).

        Returns
        -------
        Tensor
            Combined output tensor of shape (batch_size, seq_len, embed_dim).
        """

        x = ops.transpose(x, (0, 2, 1, 3)) 
        return ops.reshape(x, (ops.shape(x)[0], ops.shape(x)[1], self.embed_dim))  
    
    def call(self, query, value=None, key=None, training=False, return_attention=False):
        """
        Apply multi-head linear attention.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (batch_size, seq_len, embed_dim).
        key : Tensor, optional
            Key tensor. Defaults to `query` if None.
        value : Tensor, optional
            Value tensor. Defaults to `key` if None.
        training : bool, optional
            Whether the layer is in training mode (affects dropout).
        return_attention : bool, optional
            If True, also return the attention weights.

        Returns
        -------
        Tensor
            Output tensor after applying attention and final projection.
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