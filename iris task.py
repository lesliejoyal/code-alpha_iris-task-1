# 🌸 IRIS FLOWER CLASSIFICATION PROJECT

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📂 Load Dataset
data = pd.read_csv('/mnt/data/Iris (1).csv')

print("First 5 Rows of Dataset:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# 🎯 Separate Features and Target
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]

# ✂️ Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ⚖️ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Train Model (KNN Classifier)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 🔮 Make Predictions
y_pred = model.predict(X_test)

# 📊 Model Evaluation
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))

print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n🔲 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 🌟 Predict a New Flower Example
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print("\n🌸 Predicted Species for New Flower:", prediction[0])
