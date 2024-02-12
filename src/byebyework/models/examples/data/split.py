import os
import random

train = dict()
test = dict()
data_path = "ml-1m"

for line in open(data_path+"/ratings.dat"):
    fea = line.rstrip().split("::")
    if fea[0] not in train:
        train[fea[0]] = [line]
    elif fea[0] not in test:
        test[fea[0]] = dict()
        test[fea[0]]['time'] = int(fea[3])
        test[fea[0]]['content'] = line
    else:
        time = int(fea[3])
        if time <= test[fea[0]]['time']:
            train[fea[0]].append(line)
        else:
            train[fea[0]].append(test[fea[0]]['content'])
            test[fea[0]]['time'] = time
            test[fea[0]]['content'] = line

train_data = []
for key in train:
    for line in train[key]:
        train_data.append(line)

random.shuffle(train_data)

with open(data_path + "/train.dat", 'w') as f:
    for line in train_data:
        f.write(line)

with open(data_path + "/test.dat", 'w') as f:
    for key in test:
        f.write(test[key]['content'])