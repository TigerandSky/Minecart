import numpy as np
import tflearn

from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column = 0,
                        categorical_labels=True, n_classes=2, columns_to_ignore = [2,7])

print(data)
for p in data:
    if p[1] == 'female':
        p[1] = 1
    else:
        p[1] = 0

net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=10, batch_size = 16, show_metric= True)
print(model.predict([[3, 0, 19, 0, 0, 5.0]]))
print('nate', model.predict([[2, 0, 40, 0, 0, 80.0]]))
print('joanna', model.predict([[2,1,35,1,2,13]]))
print('dave', model.predict([[2,0,41,1,2,50]]))
print('paul', model.predict([[3,0,38,0,0,30]]))
print('bob', model.predict([[3,0,18,0,0,-185]]))
price = []
for i in data:
    price.append(i[5])
pricenp = np.array(price).astype(np.float)
print(np.mean(pricenp))