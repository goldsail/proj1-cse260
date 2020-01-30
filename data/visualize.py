#!/usr/bin/python3

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

N = 10
data = defaultdict(list)
for i in range(N):
    with open('data%d-3x12_C-30-36-32-60-108-96-840-1512-1344.txt' % i) as input:
        reader = csv.reader(input, delimiter='\t')
        for row in reader:
            data[int(row[0])].append(float(row[1]))

labels, values = data.keys(), [np.max(x) for x in data.values()]
labels, values = zip(*sorted(zip(labels, values)))
plt.plot(labels, values)
plt.xlabel('Square matrix size')
plt.ylabel('Average throughput (GF/s)')
plt.savefig('data_C.png')

with open('data.txt', 'w') as output:
    writer = csv.writer(output, delimiter='\t')
    for i in range(len(labels)):
        size = labels[i]
        speed = values[i]
        writer.writerow([size, speed])
