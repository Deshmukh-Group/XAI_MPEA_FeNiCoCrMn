import re

import numpy as np

cluster = ['tc', 'ca', 'inf', 'dt']

with open('tt', 'r') as f:
    s = f.read().rstrip('\n')

for i in cluster:
    if i in s:
        x = i

s1 = s.split(',')
list_temp = []
for i in s1:
    a = re.findall('[0-9]+', i)
    if len(a) == 1:
        m = x+ str(a[0]).zfill(3)
        list_temp.append(m)
    else:
        for j in (np.arange(int(a[0]), int(a[-1]) + 1)):
            m = x + str(j).zfill(3)
            list_temp.append(m)


print(list_temp)
with open('nodes', 'w') as f:
    for item in list_temp:
        f.write("%s\n" % item)
