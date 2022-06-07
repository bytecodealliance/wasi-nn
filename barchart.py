from asyncore import read
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

res = []

# TODO: Allow setting the directory
for resultfile in os.listdir("rust/examples/classification-example/build/RESULTS/SUMMARY"):
    f = os.path.join("rust/examples/classification-example/build/RESULTS/SUMMARY", resultfile)
# TODO: Check if its a *_all* file, we don't want to do the summary files.
    if os.path.isfile(f):
        with open(f,'r') as csvfile:
            reader = csv.reader(csvfile)
            equip = next(reader)
            backend = equip[1]
            model = equip[2]
            cpu = equip[0]
            labels = next(reader)
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                print(row)
                res.append({'threads':float(equip[3]), 'avg':float(row[2]), 'total':float(row[1]), 'sd':float(row[3])})

            plt.xlabel("Threads")
            plt.ylabel("Avg ms")

sorted_res = sorted(res, key=lambda d: d['threads'])
x = []
y = []
c = []
best=0
curr=0

for j in sorted_res:
    print("% s " %j["threads"] + " == " + "% s " %j["avg"])
    x.append(j["threads"])
    y.append(j["avg"])
    c.append('b')
    if j["avg"] < sorted_res[best]["avg"]:
        best=curr
    curr+=1

# Highlight the best result
c[best] = 'g'

plt.bar(x, y, color = c, width = .75, label = cpu + "/" + backend + "/" + model)
plt.xticks(x, rotation = 45, ha = 'right')
plt.legend(bbox_to_anchor=(0.80,-0.15), loc=1)
plt.tight_layout()
plt.savefig("barchart.png", format='png', dpi=1000)
plt.savefig("barchart.svg", format='svg')
