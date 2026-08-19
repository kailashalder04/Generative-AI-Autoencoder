# 🧠 Generative AI: Variational Autoencoder (VAE) from Scratch

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A beginner-friendly yet architecturally complete implementation of a **Variational Autoencoder (VAE)** built from scratch using TensorFlow and Keras. 

This project demonstrates how machines learn to conceptualize and generate entirely new images by compressing the Fashion MNIST dataset into a continuous 2D latent space.

---

## 📸 Project Showcase

### 1. The Latent Space Manifold
By mapping the 2D latent space, we can see how the AI smoothly morphs one clothing item into another. The items in the "in-between" spaces are entirely original creations hallucinated by the network!

![Latent Space Map](Latent%20Space%20Map.png)

### 2. Original vs. Reconstructed
The model learns to compress a 784-pixel image down to just 2 numbers, and then successfully reconstruct the image from those 2 numbers.

![Reconstruction](e)

---

## 📖 How It Works (The Simple Explanation)

A standard Autoencoder works like a game of Pictionary:
1. **The Encoder (Summarizer):** Looks at an image and compresses it into a few numbers (the "latent space").
2. **The Decoder (Artist):** Takes those numbers and tries to redraw the original image.

**The Problem:** A standard Autoencoder just memorizes specific points. The space *between* those points is empty static.
**The Solution (VAE):** A Variational Autoencoder outputs a *probability distribution* (a mean and a spread) instead of exact points. This forces the network to create a smooth, continuous landscape. Because the landscape is continuous, we can pick any random coordinate and generate a brand-new, realistic image!

---

## ⚙️ The Architecture (For the Geeks)

This model utilizes a custom training loop and the **Reparameterization Trick** to allow gradients to backpropagate through a random sampling process.

- **Encoder:** Dense network that outputs `z_mean` ($\mu$) and `z_log_var` ($\log(\sigma^2)$).
- **Sampling Layer:** Computes $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.
- **Decoder:** Dense network that reconstructs the 28x28 image from the $z$ vector.
- **Loss Function:** 
  - **Reconstruction Loss (MSE):** Ensures the generated image looks like the input.
  - **KL Divergence:** $D_{KL} = -\frac{1}{2} \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2)$. This acts as a regularizer, forcing the latent distributions to closely match a standard normal distribution, keeping the latent space densely packed and smooth.

---

## 🚀 Getting Started

Follow these steps to run the code on your own machine.

### 1. Prerequisites
You will need **Python 3.10, 3.11, or 3.12** installed (Python 3.13+ may not have stable TensorFlow support yet).

### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to prevent package conflicts.

**Windows:**
  ```bash
   python -m venv venv
   .\venv\Scripts\activate
```
***Mac/Linux:***
```Bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```Bash
pip install tensorflow matplotlib numpy
```


### 4. Run the Model
Execute the main script to train the model and generate the visual maps:

```bash
python autoencoder.py
```


---

## 📂 Project Structure
```
├── autoencoder.py      # The main VAE script (Architecture, Training, and Visualization)
├── README.md           # Project documentation
└── venv/               # (Generated) Python virtual environment

```

