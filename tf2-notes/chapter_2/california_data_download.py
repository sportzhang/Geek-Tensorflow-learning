import pickle
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)
with open(file='data/california_housing.pkl', mode='wb') as f:
    pickle.dump(housing, f)
