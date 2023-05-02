import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
data = pd.read_csv("water_potability.csv")
data = data.dropna()

X = data.drop('Potability', axis=1)
y = data['Potability']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Random Forest model and hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Model evaluation
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)



metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC Score"]
values = [accuracy, precision, recall, f1, roc_auc]

# Plot the performance metrics
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=metrics, y=values)
ax.set_ylim(0, 1)
ax.set_title("Random Forest Model Performance")
plt.show()

feature_importances = best_rf.feature_importances_
print("Feature Importances:", feature_importances)

# Assuming you have the feature importances in the following variable:
# feature_importances

features = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# Plot the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, orient="h")
ax.set_title("Feature Importance in Random Forest Model")
plt.show()
