import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading dataset...")
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# ==========================================
# 2. DEFINE THE NETWORK ARCHITECTURE
# ==========================================
latent_dim = 2  # Compress images down to just 2 coordinates (X, Y)

# The Reparameterization Layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the image."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build the Encoder
encoder_inputs = tf.keras.Input(shape=(28, 28))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Build the Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(784, activation="sigmoid")(x)
decoder_outputs = layers.Reshape((28, 28))(x)

decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")


# ==========================================
# 3. DEFINE THE VAE CLASS & LOSS FUNCTION
# ==========================================
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # 1. Encode
            z_mean, z_log_var, z = self.encoder(data)
            # 2. Decode
            reconstruction = self.decoder(z)
            
            # 3. Calculate Reconstruction Loss (MSE) manually
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=(1, 2))
            )
            # 4. Calculate KL Divergence Loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # 5. Combine for total loss
            total_loss = reconstruction_loss + kl_loss
            
        # Apply gradients to update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Display progress during training
        return {
            "loss": total_loss, 
            "reconstruction_loss": reconstruction_loss, 
            "kl_loss": kl_loss
        }


# ==========================================
# 4. COMPILE AND TRAIN
# ==========================================
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())

print("\nTraining VAE... This will take a moment!")
# Train the model (10 complete passes over the dataset)
vae.fit(x_train, epochs=10, batch_size=128)
print("\nTraining complete!\n")


# ==========================================
# 5. VISUALIZATION FUNCTIONS
# ==========================================

def plot_reconstructions(vae, x_test, n=5):
    """Plots original images from the test set vs. their AI reconstructions."""
    print("Plotting Original vs Reconstructed images...")
    
    # Run test images through the encoder to get their mean coordinates
    z_mean, _, _ = vae.encoder(x_test[:n])
    # Reconstruct images from those coordinates
    decoded_imgs = vae.decoder(z_mean).numpy()

    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("Reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.suptitle("Original vs Reconstructed (Close this window to see the next plot)")
    plt.show()


def plot_latent_space(vae, n=15, figsize=10):
    """Plots a 2D map of entirely new generated images stitched together."""
    print("Plotting the 2D Latent Space Manifold...")
    digit_size = 28
    scale = 3.0 # Scan coordinates from -3.0 to +3.0
    figure = np.zeros((digit_size * n, digit_size * n))
    
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1] 

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder(z_sample).numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("Latent Variable X")
    plt.ylabel("Latent Variable Y")
    plt.title("2D Latent Space Map (Generated Clothing)")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

# ==========================================
# 6. EXECUTE VISUALIZATIONS
# ==========================================
plot_reconstructions(vae, x_test)
plot_latent_space(vae)
