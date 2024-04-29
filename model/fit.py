import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_df = pd.read_csv("IRIS (1).csv")
X = iris_df.drop(["species"], axis=1)
Y = iris_df["species"]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3, random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
with open('Iris_pickle_file', 'wb') as pkl:
    pickle.dump(model, pkl)
