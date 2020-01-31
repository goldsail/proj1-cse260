#!/usr/bin/python3

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

N = 1
data_naive = defaultdict(list)
data_blocked = defaultdict(list)
data_blas = defaultdict(list)
for i in range(N):
    with open('naive-data.txt' % i) as input:
        reader = csv.reader(input, delimiter='\t')
        next(reader)
        next(reader)
        for row in reader:
            data_naive[int(row[0].split(' ')[1])].append(float(row[1].split(' ')[1]))
    with open('blocked-data.txt' % i) as input:
        reader = csv.reader(input, delimiter='\t')
        next(reader)
        next(reader)
        for row in reader:
            data_blocked[int(row[0].split(' ')[1])].append(float(row[1].split(' ')[1]))
    with open('blas-data.txt' % i) as input:
        reader = csv.reader(input, delimiter='\t')
        next(reader)
        next(reader)
        for row in reader:
            data_blas[int(row[0].split(' ')[1])].append(float(row[1].split(' ')[1]))


labels_naive, values_naive = data_naive.keys(), [np.max(x) for x in data_naive.values()]
labels_naive, values_naive = zip(*sorted(zip(labels_naive, values_naive)))
labels_blocked, values_blocked = data_blocked.keys(), [np.max(x) for x in data_blocked.values()]
labels_blocked, values_blocked = zip(*sorted(zip(labels_blocked, values_blocked)))
labels_blas, values_blas = data_blas.keys(), [np.max(x) for x in data_blas.values()]
labels_blas, values_blas = zip(*sorted(zip(labels_blas, values_blas)))

f,ax = plt.subplots()
naive = plt.scatter(labels_naive, values_naive, color='b', alpha=0.5)
blocked = plt.scatter(labels_blocked, values_blocked, color='r', alpha=0.5)
blas = plt.scatter(labels_blas, values_blas, color='g', alpha=0.5)
plt.xlabel('Square matrix size')
plt.xscale('log')
ax.set_xticks([2**n for n in range(5,12)])
ax.set_xticklabels([2**n for n in range(5,12)])
plt.ylabel('Average throughput (GF/s)')
plt.legend((naive, blocked, blas), ('Naive', 'Blocked', 'Blas'))
plt.savefig('plot.png')

if False:
    with open('data.txt', 'w') as output:
        writer = csv.writer(output, delimiter='\t')
        for i in range(len(labels)):
            size = labels[i]
            speed = values[i]
            writer.writerow([size, speed])
