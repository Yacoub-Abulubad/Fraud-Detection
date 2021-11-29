#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard



#Imagine a specific fraudulent usage pattern
#(high conversion rates, high revenues per complete, or something like this).
#What would be the basic approach to detect these users via, for example, ML?
# %%
#Load in the datasets
events = pd.read_csv('events_duration_na.csv')
events = events.loc[:, ~events.columns.str.contains('^Unnamed')]
events = events.loc[:, ~events.columns.str.contains('created_at')]
#%%
train_x_1, test_x_1 = train_test_split(events, test_size=0.2, shuffle=False)
train_x = train_x_1.values
test_x = test_x_1.values
#%%
input_layer = Input(shape=(2, ))
encoder = Dense(2, activation='relu', activity_regularizer=regularizers.l1(0.000001))(input_layer)
encoder = Dense(8, activation='relu')(encoder)
encoder = Dense(4, activation='relu')(encoder)
decoder = Dense(4, activation='relu')(encoder)
decoder = Dense(8, activation='relu')(encoder)
decoder = Dense(2)(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
#%%
autoencoder.compile(metrics=['accuracy'], loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.00001))
#%%
cp = ModelCheckpoint(filepath="BitBurst_autoencoder_na4.h5", save_best_only=False, verbose=0)
#%%
history = autoencoder.fit(train_x, train_x, epochs=50, shuffle=False, validation_data=(test_x,test_x), verbose=1, callbacks=cp).history
#%%
plt.plot(history['loss'][2:], linewidth=2, label='Train')
plt.plot(history['val_loss'][2:], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


#%%
test_x_predictions = autoencoder.predict(test_x)
mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse})
error_df.describe()


# %%
test_x_predictions = pd.DataFrame(test_x_predictions, columns=['revenue','duration'])
sns.regplot(y="revenue", x="duration", data=test_x_1)
sns.regplot(y="revenue", x="duration", data=test_x_predictions)


#%%
plt.plot(test_x_predictions)
plt.show()
plt.plot(test_x)
plt.show()
#%%
