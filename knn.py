import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


data = pd.read_csv("Iris.csv")

print("First few rows of the dataset:")
print(data.head())

X = data.iloc[:, :4]
y = data.iloc[:, -1]

print("\nFeature data (first 5 rows):")
print(X.head())

print("\nLabels (first 5 rows):")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20
)

print("\nTraining features (first 5 rows):")
print(X_train.head())

print("\nTesting features (first 5 rows):")
print(X_test.head())


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

print("\nArray:", y_pred)
print("\nActual labels:", y_test.values)


cm = confusion_matrix(y_test, y_pred)


ac = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", ac)