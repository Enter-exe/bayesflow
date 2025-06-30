import keras
from keras import ops, random
from keras.layers import Layer
from keras.layers import Dropout


class LinearAttention(Layer):
    """
    lab.py

    Defines a custom Keras layer implementing linear attention for use in deep learning models.
    This layer projects the input into query, key, and value spaces and computes attention
    using a simplified linear formulation instead of traditional scaled dot-product attention.

    Classes:
        LinearAttention: Implements linear attention with learnable projections and optional bias.
    """
    
    
    def __init__(self, feature_dim, dropout_rate=0.2, use_bias=False, **kwargs):
        """
        Initializes the LinearAttention layer.

        Args:
            feature_dim (int): Dimensionality of query/key/value projections.
            dropout_rate (float): Dropout rate for attention weights.
            use_bias (bool): Whether to include a bias term in projections.
            **kwargs: Additional keyword arguments for base Layer.
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.dropout = Dropout(dropout_rate)
        self.use_bias = use_bias

    def build(self, input_shape):
        """
        Creates learnable weights for query, key, and value projections.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.query_weights = self.add_weight(
            shape=(input_shape[-1], self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="query_weights"
        )

        self.key_weights = self.add_weight(
            shape=(input_shape[-1], self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="key_weights"
        )

        self.value_weights = self.add_weight(
            shape=(input_shape[-1], self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="value_weights"
        )

        if self.use_bias:
            self.query_bias = self.add_weight(
                shape=(self.feature_dim,),
                initializer="zeros",
                trainable=True,
                name="query_bias",
            )
            self.key_bias = self.add_weight(
                shape=(self.feature_dim,),
                initializer="zeros",
                trainable=True,
                name="key_bias",
            )
            self.value_bias = self.add_weight(
                shape=(self.feature_dim,),
                initializer="zeros",
                trainable=True,
                name="value_bias",
            )

        super().build(input_shape)

    def call(self, query, key=None, value=None, training=False, return_attention=False):
        """
        Computes the forward pass for the linear attention mechanism.

        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (Tensor, optional): Key tensor. Defaults to `query` if None.
            value (Tensor, optional): Value tensor. Defaults to `key` if None.
            training (bool): Whether the layer is in training mode (affects dropout).
            return_attention (bool): If True, also return the attention weights.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Output tensor of shape 
            (batch_size, seq_len, feature_dim), optionally with attention weights.
        """
        
        if key is None:
            key = query
        if value is None:
            value = key
        
        queries = ops.matmul(query, self.query_weights)
        keys = ops.matmul(key, self.key_weights)
        values = ops.matmul(value, self.value_weights)
        if self.use_bias:
            queries = queries + self.query_bias
            keys = keys + self.key_bias
            values = values + self.value_bias
        
        queries = ops.relu(queries) + 1
        keys = ops.relu(keys) + 1

        if training: values = self.dropout(values, training=training)
      
        keys_t = ops.transpose(keys, axes=(0, 2, 1))    
        numerator = ops.matmul(keys_t, values)          
        denominator = ops.sum(keys, axis=-2, keepdims=True)  

        context = numerator / (denominator + 1e-6)  

        if return_attention:
            
            return context, keys

        queries = ops.transpose(queries, (0,2,1))
        output = ops.matmul(context, queries)
        return output

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.

        Returns:
            TensorShape: Shape of the output tensor.
        """
        return (input_shape[0], self.feature_dim)
    
    def get_config(self):
        """
        Returns the config of the layer for serialization.

        Returns:
            dict: Configuration dictionary.
        """

        config = super().getconfig()
        config.update({
            "feature_dim": self.feature_dim,
            "droupout_rate": self.dropout.rate,
            "use_bias": self.use_bias
        })
        return config
