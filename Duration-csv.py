#%%
import pandas as pd
import numpy as np

#Load in the datasets
events = pd.read_csv('events.csv')
events = events[['app_id','uid','survey_id','revenue','type','platform','country','created_at']]
#users = pd.read_csv('users.csv')
events['created_at'] = events['created_at'].map(lambda x: x.split('T')[1])
events['created_at'] = events['created_at'].map(lambda x: x.split('.')[0])

events['created_at'] = pd.to_timedelta(events['created_at'])

events['revenue'] = events['revenue'].fillna(0)


max_a = events.created_at.max()
min_a = events.created_at.min()
min_norm = 0
max_norm = 1
events['created_at'] = (events.created_at- min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm
events['duration'] = np.nan
#%%
created_time = 0
complete_time = 0
i = len(events.values)
for i in range(len(events.values)-1,0,-1):
    if events['type'][i] == "complete":
        uid = events['uid'][i]
        sid = events['survey_id'][i]
        complete_time = events['created_at'][i]
        c = i
        for c in range (c,0,-1):
            if events['type'][c] == "open" and events['uid'][c] == uid and events['survey_id'][c] == sid:
                events['duration'][i] = complete_time - events['created_at'][c]
                break
#%%
events.drop(['app_id','type','platform','country'],axis='columns',inplace=True)
max_a = events.duration.max()
min_a = events.duration.min()
events['duration'] = (events.duration- min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm
max_a = events.revenue.max()
min_a = events.revenue.min()
events['revenue'] = (events.revenue- min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm

#%%
events.to_csv('events_duration.csv')
#%%
events = events[events['duration'].notna()]
events = events.loc[:, ~events.columns.str.contains('^Unnamed')]
#%%
events.to_csv('events_duration_na.csv')


#%%

