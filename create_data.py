import pandas as pd
from sklearn.model_selection import train_test_split


audi = pd.read_csv('data/audi.csv')

x = audi.drop('price', axis=1)
y = audi['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=428, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=640, random_state=0)

x_train['price'] = y_train
x_val['price'] = y_val
x_test['price'] = y_test

pd.DataFrame(x_train).to_csv('./data/train.csv', index=False)
pd.DataFrame(x_val).to_csv('./data/val.csv', index=False)
pd.DataFrame(x_test).to_csv('./data/test.csv', index=False)
