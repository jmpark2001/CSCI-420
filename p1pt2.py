import json
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('layer0_paper.json')
df = df[['Id', 'Ti', 'D', 'AA']]
print(df)
for x, rows in df.iterrows():
    df['AA'][x] = ';'.join([authors['AuN'] for authors in df['AA'][x]])
print(df)
df = df.rename(columns={'Id': 'paperID', 'Ti': 'paper title', 'D': 'publish years', 'AA': 'authors'})
print(df)
compression_options = dict(method='zip', archive_name='output2.csv')
df.to_csv('output2.zip', compression=compression_options)

f = open('layer0_paper.json')
micro = json.load(f)
normalized_authors = pd.json_normalize(micro, record_path=['AA'])
print(normalized_authors)
normalized_authors = normalized_authors.groupby(normalized_authors['AuN']).size().reset_index(name='count')
normalized_authors = normalized_authors.sort_values(by=['count'], ascending=False).head(10)
print(normalized_authors)

graph = normalized_authors.plot.bar(x='AuN', y='count', rot=0, figsize=[10, 5])
plt.xticks(fontsize=6)
graph.set_xlabel('Author Names')
graph.set_ylabel('Number of Papers')
plt.show()