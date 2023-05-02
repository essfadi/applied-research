import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("water_potability.csv")

# Plot the distribution of key water quality parameters
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
sns.boxplot(data=data, y="ph", ax=axes[0, 0])
sns.boxplot(data=data, y="Hardness", ax=axes[0, 1])
sns.boxplot(data=data, y="Solids", ax=axes[0, 2])
sns.boxplot(data=data, y="Chloramines", ax=axes[1, 0])
sns.boxplot(data=data, y="Sulfate", ax=axes[1, 1])
sns.boxplot(data=data, y="Conductivity", ax=axes[1, 2])
sns.boxplot(data=data, y="Organic_carbon", ax=axes[2, 0])
sns.boxplot(data=data, y="Trihalomethanes", ax=axes[2, 1])
sns.boxplot(data=data, y="Turbidity", ax=axes[2, 2])
fig.suptitle("Distribution of Key Water Quality Parameters")
plt.show()
