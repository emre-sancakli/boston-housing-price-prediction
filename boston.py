import numpy as np
# load the dataset
dataset = np.loadtxt('housing.csv')
dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split
# split the dataset into training and test datasets.
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.20)


from sklearn.preprocessing import MinMaxScaler
# apply min-max scaling
mms = MinMaxScaler()
mms.fit(training_dataset_x)
training_dataset_x = mms.transform(training_dataset_x)
test_dataset_x = mms.transform(test_dataset_x)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# create keras model and add the layers
model = Sequential(name='BostonHousingPrices')
model.add(Dense(100, input_dim = training_dataset_x.shape[1], activation='relu', name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))

# compile the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# fit the model and assign it to 'hist' variable
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)

# this part is for drawing the graphs
import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Mean Absolute Error - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.plot(range(1, len(hist.history['mae']) + 1), hist.history['mae'])
plt.plot(range(1, len(hist.history['val_mae']) + 1), hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()


# testing the model with using evaluate method
test_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(test_result)):
    print(f'{model.metrics_names[i]} ---> {test_result[i]}')

"""
0.63796   0.00   8.140  0  0.5380  6.0960  84.50  4.4619   4  307.0  21.00 380.02  10.26 -->  18.20
1.05393   0.00   8.140  0  0.5380  5.9350  29.30  4.4986   4  307.0  21.00 386.85   6.58 --> 23.10
"""


# creating simple data group for the estimation part then predicting the results
predict_data = np.array([0.63796, 0.00, 8.140, 0, 0.5380, 6.0960, 84.50, 4.4619, 4, 307.0, 21.00, 380.02, 10.26])
predict_data = mms.transform(predict_data.reshape(1, -1))
predict_result = model.predict(predict_data)
print(f'Predicted result: {predict_result[0, 0]}')
