import matplotlib.pyplot as plt
import numpy as np

# Models
models = [
    "Logistic Regression",
    "Linear SVM",
    "Kernel SVM",
    "Naïve Bayes",
    "CNN",
    "LSTM",
    "Transformer"
]

# Training Complexity (Big-O notation, converted to approximate scaling factors for visualization)
training_complexity = [
    2,  # O(n * d * k) - Similar to Linear SVM
    2,  # O(n * d * k) - Similar to Logistic Regression
    8,  # O(n^2 * d) - Expensive training
    1,  # O(n * d) - Very fast
    4,  # O(L * n * d * f * k) - CNNs have moderate complexity
    6,  # O(L * n * d * h^2) - LSTMs are slower than CNNs
    10  # O(L * n^2 * d) - Transformers are the most computationally expensive
]

# Inference Complexity (Big-O notation, converted to approximate scaling factors for visualization)
inference_complexity = [
    1,  # O(d) - Very fast
    1,  # O(d) - Similar to Logistic Regression
    5,  # O(s * d) - Slower due to kernel computation
    1,  # O(d) - Very efficient
    3,  # O(n * d * f * k) - CNNs are efficient
    6,  # O(L * n * d * h^2) - LSTMs are slow due to sequential computation
    8  # O(L * n^2 * d) - Transformers are slower for long texts
]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.4
indices = np.arange(len(models))

ax.barh(indices - bar_width/2, training_complexity, bar_width, label="Training Complexity")
ax.barh(indices + bar_width/2, inference_complexity, bar_width, label="Inference Complexity")

ax.set_xlabel("Relative Computational Complexity (Higher = Slower)")
ax.set_title("Computational Complexity of ML and Neural Models (NLP)")
ax.set_yticks(indices)
ax.set_yticklabels(models)
ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()