import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


df = pd.read_csv('Dataset.csv')


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Quality_Label', palette='mako')
plt.title("Software Quality Label Distribution")
plt.show()


# Handling null values
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


# Encoding categorical values
le = LabelEncoder()
df['Has_Unit_Tests'] = le.fit_transform(df['Has_Unit_Tests'])
df['Quality_Label'] = le.fit_transform(df['Quality_Label'])


# Feature scaling
X = df.drop('Quality_Label', axis=1)
y = df['Quality_Label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


# Neural Network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation="softmax")
])

nn_model.compile(optimizer=Adam(0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=20, verbose=0)


# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)


models = ['Logistic Regression', 'Naive Bayes', 'Neural Network']
preds = [lr_model.predict(X_test), nb_model.predict(X_test), np.argmax(nn_model.predict(X_test), axis=1)]
probs = [lr_model.predict_proba(X_test), nb_model.predict_proba(X_test), nn_model.predict(X_test)]

accs = [accuracy_score(y_test, p) for p in preds]
f1s = [f1_score(y_test, p, average='weighted') for p in preds]


# Bar Chart
x = np.arange(len(models))
plt.bar(x - 0.2, accs, 0.4, label='Accuracy')
plt.bar(x + 0.2, f1s, 0.4, label='F1 Score')
plt.xticks(x, models)
plt.title("Performance Comparison")
plt.legend()
plt.show()


# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[i], ax=ax, cmap='Purples')
    ax.set_title(models[i])
plt.show()


# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
plt.figure(figsize=(8, 6))
for i in range(len(models)):
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs[i].ravel())
    plt.plot(fpr, tpr, label=f'{models[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.legend()
plt.show()

from google.colab import drive
drive.mount('/content/drive')