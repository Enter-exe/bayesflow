import keras
from keras import ops, random
from keras.layers import Layer
from keras.layers import Dropout


class LinearAttention(Layer):
    def __init__(self, feature_dim, dropout_rate=0.2, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.dropout = Dropout(dropout_rate)
        self.use_bias = use_bias

    def build(self, input_shape):
        # Learnable weight matrices for queries, keys, and values
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

    def call(self, inputs, training=False, return_attention=False):
        # Compute queries, keys, and values
        queries = ops.matmul(inputs, self.query_weights)
        keys = ops.matmul(inputs, self.key_weights)
        values = ops.matmul(inputs, self.value_weights)

        if self.use_bias:
            queries = queries + self.query_bias
            keys = keys + self.query_bias
            values = values + self.query_bias
        
        queries = ops.relu(queries) + 1
        keys = ops.relu(keys) + 1

        # Apply Dropout to values during training
        if training: values = self.dropout(values, training=training)

        # Compute the numerator and denominator for linear attention
        #numerator = ops.matmul(keys, values, transpose_a=True)  # Weighted sum of values

      
        keys_t = ops.transpose(keys, axes=(0, 2, 1))     # shape: (batch, feature_dim, seq_len)
        numerator = ops.matmul(keys_t, values)           # result: (batch, feature_dim, feature_dim)
        denominator = ops.sum(keys, axis=-2, keepdims=True)  # Sum of keys

        # Compute context by dividing numerator by denominator
        context = numerator / (denominator + 1e-6)  # Add small epsilon for numerical stability

        if return_attention:
            # Optional: return keys as attention scores for interpretability
            return context, keys

        # Optionally weight the context with queries (not always necessary in linear attention)
        queries = ops.transpose(queries, (0,2,1))
        output = ops.matmul(context, queries)
        #return ops.squeeze(output, axis=-2)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.feature_dim)
    
    def get_config(self):
        config = super().getconfig()
        config.update({
            "feature_dim": self.feature_dim,
            "droupout_rate": self.dropout.rate,
            "use_bias": self.use_bias
        })
        return config
