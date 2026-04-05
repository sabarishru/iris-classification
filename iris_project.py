import pandas as pd

df = pd.read_csv("C:/Users/SABARISH R/Downloads/archive/Iris.csv")


if "Id" in df.columns:
    df = df.drop("Id", axis=1)

X = df.drop("Species", axis=1)
y = df["Species"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_pred))

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))
new_data = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
result = model.predict(new_data)

print("Predicted Species:", result[0])