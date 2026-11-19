# NumPy Core Operations & Math Foundations

This repository contains my completed NumPy exercises (NP-E1 to NP-M4). It covers fundamental array manipulations, vector/matrix operations, cosine similarity, top-K retrieval, pairwise distances, and a simple multi-layer neural network forward pass.

##  Requirements
Ensure you have the following environment set up:
- Python 3.8+
- NumPy (`pip install numpy`)

##  How to Run
To reproduce the results, please follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GWX123555/numpy.homework.git
   cd numpy.homework
 Run the verification script:
Ensure the provided helpers.py, the model weights, and the check.py script are in the same directory as image_encoder.py, then run:

bash
python check.py
 Expected Output
When you run the verification script, all components should pass. The terminal output should look exactly like this:

text
assets      PASS
preprocess  PASS
patches     PASS
tokens      PASS
block0      PASS
last_tokens PASS
output      PASS
 Implementation Highlights
Strict Float32 Precision: A major challenge in this project was preventing NumPy from upcasting arrays to float64. I explicitly cast all model weights, intermediate scaling factors (np.sqrt), and final outputs to float32 to pass the precision checks.

Correct Patchify Order: In the patchify method, the images are reshaped and transposed using x.transpose(0, 2, 4, 1, 3, 5). This ensures the patches are extracted sequentially from left-to-right and top-to-bottom, which is crucial for preserving spatial features.

Efficient Attention Mechanism: The MultiHeadAttention class efficiently splits, scales, and merges heads without relying on slow Python loops, utilizing matrix multiplications (@) and broadcasting for the Q, K, and V computations.
