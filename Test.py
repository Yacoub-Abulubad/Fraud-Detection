#%%
import keras
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette
from tensorflow.python.tf2 import enable
#%%
reconstructed_model = keras.models.load_model("BitBurst_autoencoder_na4.h5")
#%%
events = pd.read_csv('events_duration_na.csv')
events = events.loc[:, ~events.columns.str.contains('^Unnamed')]
events = events.loc[:, ~events.columns.str.contains('created_at')]
events = events.loc[:, ~events.columns.str.contains('uid')]
events = events.loc[:, ~events.columns.str.contains('survey_id')]
train_x_1 = events[:int(len(events.values)*0.8)]
test_x = events[-int(len(events.values)*0.2):]
test_x = test_x.reset_index()
test_x_1 = test_x.loc[:, ~test_x.columns.str.contains('index')]
train_x = train_x_1.values
test_x = test_x_1.values

#%%
test_x_predictions = reconstructed_model.predict(test_x)
mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse})
error_df.describe()

# %%
x_train_pred = reconstructed_model.predict(train_x)
train_mae_loss = np.mean(np.abs(x_train_pred - train_x), axis=1)    
threshold = np.median(train_mae_loss)+iqr(train_mae_loss)*1.5
print("Reconstruction error threshold: ", threshold)
anomaly = error_df.query('Reconstruction_error > 0.03667132036205529')

g = sns.scatterplot(y=error_df.Reconstruction_error,x=error_df.index, alpha=0.8, label="Normal")
sns.scatterplot(y=anomaly.Reconstruction_error,x=anomaly.index, color='red', label="Fraud")
plt.legend()
ax1 = g.axes
g.set(xlabel="Entries")
ax1.axhline(threshold, ls='--',c='green')
sns.despine()

plt.savefig("Fraud_fig")
#%%
anomalies = error_df['Reconstruction_error'] > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
locations = np.where(anomalies)
#%%
locations = np.transpose(locations)
loc_index = pd.DataFrame(locations)
#%%
events = pd.read_csv('events_duration_na.csv')
events = events.loc[:, ~events.columns.str.contains('^Unnamed')]
events = events.loc[:, ~events.columns.str.contains('created_at')]
p_anomaly_data = events[-int(len(events.values)*0.2):]
p_anomaly_data = p_anomaly_data.reset_index()
p_anomaly_data = p_anomaly_data.loc[:, ~p_anomaly_data.columns.str.contains('index')]
# %%
anomaly_data = pd.DataFrame()
for index in loc_index[0]:
    anomaly_data = anomaly_data.append(p_anomaly_data.iloc[index])

anomaly_data = anomaly_data.loc[:, ~anomaly_data.columns.str.contains('duration')]
anomaly_data = anomaly_data.loc[:, ~anomaly_data.columns.str.contains('revenue')]
#%%
anomaly_data.to_csv("Anomaly_data.csv")
