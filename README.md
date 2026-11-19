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
   python "numpy.homework.py"
    Expected Output
When you run the script, you will see the outputs for all 12 exercises sequentially. Below is a summary of the expected output:

NP-E1 to NP-E8: Array properties, slicing, copying, element-wise operations, dot products, vector norms, angles, distances, unit vectors, and broadcasting.

M1: Calculates the angle between a query vector and an array of vectors.

M2: Retrieves the top-3 most similar images based on cosine similarity, correctly handling zero vectors.

M3: Computes the pairwise Euclidean distance matrix between two sets of points.

M4: Performs a forward pass for a 2-layer neural network with ReLU activation.

 Implementation Highlights
Robustness: Functions like angle_deg, unit_vector, and top_k_cosine include strict type and dimension checking. They raise ValueError when encountering zero vectors to prevent division-by-zero errors.

Efficient Matrix Operations: Used np.newaxis and broadcasting in pairwise_dist to compute distance matrices without using slow Python loops.

Top-K Selection: Implemented top_k_cosine using boolean masking to filter out zero vectors, followed by np.argsort to efficiently find the highest similarity scores.


