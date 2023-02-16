import pandas as pd



def euclidean(list1, list2):
    ed = 0
    for i in range(len(list1)):
        ed += (list1[i] - list2[i])**2
    ed = ed**.5
    return ed


k = 50
df = pd.read_csv('KNN_train.csv')
df2 = pd.read_csv('KNN_test.csv')
all_rows = []
for index, rows in df.iterrows():
    row_list = [rows.SepalLengthCm, rows.SepalWidthCm, rows.PetalLengthCm, rows.PetalWidthCm, rows.Labels]
    all_rows.append(row_list)

valid_rows = []
for index, rows in df2.iterrows():
    row_list = [rows.SepalLengthCm, rows.SepalWidthCm, rows.PetalLengthCm, rows.PetalWidthCm]
    valid_rows.append(row_list)

all_distances = []
for i in valid_rows:
    distances = []
    for j in all_rows:
        distances.append([euclidean(i, j), j[4]])
    all_distances.append(distances)

sorted_list = []
for i in all_distances:
    i = sorted(i)
    sorted_list.append(i[:k])

final_dicts = []
for i in sorted_list:
    results = {}
    for j in i:
        if j[1] not in results:
            results[j[1]] = 1
        else:
            results[j[1]] += 1
    final_dicts.append(results)

final_labels = []
for i in final_dicts:
    final_labels.append(max(i, key=i.get))

#print(final_labels)

df2['Labels'] = final_labels
print(df2)
compression_options = dict(method='zip', archive_name='KNN_Test.csv')
df2.to_csv('KNN_Test.zip', compression=compression_options)