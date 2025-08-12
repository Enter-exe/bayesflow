import keras
from keras import ops, random
from keras.layers import Layer
from keras.layers import Dropout

@serializable("bayesflow.networks")
class LinearAttention(Layer):
    """
    A custom Keras layer implementing linear attention.

    This layer projects input into query, key, and value spaces and computes attention
    using a simplified linear formulation instead of traditional scaled dot-product attention.
    """

    
    
    def __init__(self, feature_dim, dropout_rate=0.2, use_bias=False, **kwargs):
        """
        Initialize the LinearAttention layer.

        Parameters
        ----------
        feature_dim : int
            Dimensionality of query/key/value projections.
        dropout_rate : float, optional
            Dropout rate for attention weights (default is 0.2).
        use_bias : bool, optional
            Whether to include a bias term in projections (default is False).
        **kwargs : dict
            Additional keyword arguments for the base Layer.
        """

        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.dropout = Dropout(dropout_rate)
        self.use_bias = use_bias

    def build(self, input_shape):
        """
        Create learnable weights for query, key, and value projections.

        Parameters
        ----------
        input_shape : TensorShape
            Shape of the input tensor.
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
        Compute the forward pass for the linear attention mechanism.

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
        Tensor or Tuple[Tensor, Tensor]
            Output tensor of shape (batch_size, seq_len, feature_dim),
            optionally with attention weights.
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
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : TensorShape
            Shape of the input tensor.

        Returns
        -------
        TensorShape
            Shape of the output tensor.
        """

        return (input_shape[0], self.feature_dim)
    
    def get_config(self):
        """
        Return the config of the layer for serialization.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().getconfig()
        config.update({
            "feature_dim": self.feature_dim,
            "droupout_rate": self.dropout.rate,
            "use_bias": self.use_bias
        })
        return config
