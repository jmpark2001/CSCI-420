from bz2 import compress
from ssl import ALERT_DESCRIPTION_ACCESS_DENIED
from time import asctime
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('crypto_github_events.csv')
df = df.groupby(by=['userID'])['actions'].count().reset_index()
df = df.sort_values('actions', ascending=False).head(10)
top_users = df['userID'].tolist()
df2 = pd.read_csv('crypto_github_events.csv')
df2 = df2.loc[df2['userID'].isin(top_users)]
df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.groupby([df2['date'].dt.year.rename('year'), df2['date'].dt.month.rename('month'), 'userID'])['actions'].count().reset_index()
df2['date'] = pd.to_datetime(df2.year.astype(str) + '/' + df2.month.astype(str))
df2['date'] = df2['date'].dt.strftime('%m/%Y')
df2 = df2.drop('year', 1)
df2 = df2.drop('month', 1)
df2.date = pd.Categorical(df2.date, categories=df2.date.unique(), ordered=True)
df2 = df2.pivot(index=('userID'), columns='date', values=['actions'])
df2['actions'] = df2['actions'].fillna(0).astype(int)
df2.columns = df2.columns.droplevel()
df2 = df2.rename_axis(None, axis=1)
compression_options = dict(method='zip', archive_name='output.csv')
df2.to_csv('output.zip', compression=compression_options)
print(df2)
df3 = pd.read_csv('crypto_github_events.csv')
df3['date'] = pd.to_datetime(df3['date'])
df3 = df3.groupby([df3['date'].dt.year.rename('year'), 'userID'])['actions'].count().reset_index()
df3 = df3.loc[df3['year'] == 2016]
df3 = df3.sort_values('actions', ascending=False).head(5)
top_users = df3['userID'].tolist()
df4 = pd.read_csv('crypto_github_events.csv')
df4 = df4.loc[df4['userID'].isin(top_users)]
df4['date'] = pd.to_datetime(df4['date'])
df4 = df4.groupby([df4['date'].dt.year.rename('year'), df4['date'].dt.month.rename('month'), 'userID'])['actions'].count().reset_index()
df4 = df4.loc[df4['year'] == 2016]
df4['date'] = pd.to_datetime(df4.year.astype(str) + '/' + df4.month.astype(str))
df4['date'] = df4['date'].dt.strftime('%m/%Y')
df4 = df4.drop('year', 1)
df4 = df4.drop('month', 1)
user_count = 0
df5 = df4.loc[df4['userID'] == top_users[user_count]].reset_index(drop=True)
user_count += 1
df6 = df4.loc[df4['userID'] == top_users[user_count]].reset_index(drop=True)
user_count += 1
df7 = df4.loc[df4['userID'] == top_users[user_count]].reset_index(drop=True)
user_count += 1
df8 = df4.loc[df4['userID'] == top_users[user_count]].reset_index(drop=True)
user_count += 1
df9 = df4.loc[df4['userID'] == top_users[user_count]].reset_index(drop=True)
print(df5)
print(df6)
print(df7)
print(df8)
print(df9)
figure = plt.figure(figsize=(10,5))
for dataframe in [df5, df6, df7, df8, df9]:
    plt.plot(dataframe['date'], dataframe['actions'], label=dataframe['userID'][0])
figure.legend(loc='upper left', prop={'size': 7})
plt.show()