import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model

def build_mlp_feature_map(input_shape, num_tokens=64, lm_hidden_dim=768):
    """
    Creates a Multilayer Perceptron model that outputs a feature map with 64 visual tokens
    and maps them to the language model space using a linear layer.
    
    Args:
        input_shape: Tuple representing the input feature map dimensions.
        num_tokens: Number of visual tokens to output.
        lm_hidden_dim: Hidden dimension of the language model.

    Returns:
        A tf.keras.Model object representing the MLP network. 
    """
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Flatten input to a 1D vector
    flattened = tf.keras.layers.Flatten()(inputs)
    
    # Hidden layers (adjust neuron counts as needed)
    hidden1 = tf.keras.layers.Dense(128, activation='relu')(flattened)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    
    # Output layer with no activation, producing the feature map
    visual_tokens = tf.keras.layers.Dense(num_tokens)(hidden2)
    
    # Reshape output to desired feature map shape
    visual_tokens = tf.keras.layers.Reshape((int(tf.math.sqrt(num_tokens)), int(tf.math.sqrt(num_tokens))))(visual_tokens)
    
    # Flatten the visual tokens to pass through a linear layer
    flattened_visual_tokens = tf.keras.layers.Flatten()(visual_tokens)
    
    # Linear layer to map visual tokens to language model space
    lm_tokens = tf.keras.layers.Dense(lm_hidden_dim)(flattened_visual_tokens)
    
    model = tf.keras.Model(inputs=inputs, outputs=lm_tokens)
    return model

# Example usage
input_shape = (224, 224)
num_tokens = 64
lm_hidden_dim = 768  # Hidden dimension of GPT-2

# Build the MLP model
model = build_mlp_feature_map(input_shape, num_tokens, lm_hidden_dim)
model.summary()

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example input image
import numpy as np
input_image = np.random.rand(1, 224, 224).astype(np.float32)

# Get the visual tokens from the MLP model
visual_tokens = model.predict(input_image)

# Convert visual tokens to language tokens using the tokenizer
language_tokens = tokenizer.encode(visual_tokens.flatten().tolist(), return_tensors='tf')

print("Language tokens:", language_tokens)