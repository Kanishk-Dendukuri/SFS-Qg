from minepy import MINE
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Y = np.array([2, 5, 7, 9, 10, 12, 14, 15])

# Create a MINE object
mine = MINE()

# Compute MIC
mine.compute_score(X, Y)
print("MIC:", mine.mic())
