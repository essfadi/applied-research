import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv("water_potability.csv")

# Calculate the correlation matrix
corr_matrix = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot the correlation matrix
sns.set(style="white")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True, fmt=".2f")
ax.set_title("Correlation Matrix of Water Quality Parameters")
plt.show()
