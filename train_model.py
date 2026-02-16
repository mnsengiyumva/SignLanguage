import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load your 221 rows of data
df = pd.read_csv('hand_data.csv')

# 2. Split into Features (X - the coordinates) and Labels (y - the sign names)
X = df.drop('label', axis=1)
y = df['label']

# 3. Split into Training (80%) and Testing (20%) sets
# This helps us see if the AI actually learned or just memorized
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 4. Train the Model
print("Training the brain... this should be fast!")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Check the "Grade" (Accuracy)
y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# 6. Save the model to a file so we can use it in the real app
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Brain saved as model.p!")