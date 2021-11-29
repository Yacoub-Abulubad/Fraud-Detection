import json
import pandas as pd

#We import the users json file 
f = open('users.json')
data = json.load(f)

#We list and insert the columns of the user json file
df = pd.DataFrame(data, columns=['app_id', 'uid'])

#Here I had to manually extract the nested data under 'user_data' 
#since it was not properly coded 
#(The json file had the nested data as a string)
df1 = pd.DataFrame()
for i in range(0, len(data)):

    #We run it through a loop which loads the nested data,
    #previously a string, as a json file and we append it to our dataframe
    df1 = df1.append(json.loads(data[i]['user_data']), ignore_index=True)

#finally we concatenate both dataframes into the variable user_data 
#to convert it into a .csv format and save it
user_data = pd.concat([df,df1], axis=1)
user_data.to_csv('users.csv')