import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []

def moving_avg(y, N=30):
    return np.convolve(y, np.ones((N,))/N, mode='valid')

# TODO: Allow setting the directory
for resultfile in os.listdir("rust/examples/classification-example/build/RESULTS"):
    f = os.path.join("rust/examples/classification-example/build/RESULTS", resultfile)
# TODO: Check if its a *_all* file, we don't want to do the summary files.
    if os.path.isfile(f):
        with open(f,'r') as csvfile:
            reader = csv.reader(csvfile)
            equip = next(reader)
            backend = equip[1]
            model = equip[2]
            cpu = equip[0]
            threads = equip[3]
            labels = next(reader)
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                x.append(float(row[0]))
                y.append(float(row[1]))

            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            x=x[0:75]
            y=y[0:75]

            plt.plot(x, y, linestyle = 'solid',
                marker = 'o',label = cpu + "/" + backend + "/" + model + "/" + threads + " threads")

            plt.xticks(rotation = 45, ha = 'right')
            x.clear()
            y.clear()

plt.legend(bbox_to_anchor=(0.80,-0.15), loc=1)
plt.tight_layout()
plt.savefig("linechart.png", format='png', dpi=1000)
plt.savefig("linechart.svg", format='svg')