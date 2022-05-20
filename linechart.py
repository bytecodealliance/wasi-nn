import os
import matplotlib.pyplot as plt
import csv

x = []
y = []

# TODO: Allow setting the directory
for resultfile in os.listdir("RESULTS"):
    f = os.path.join("RESULTS", resultfile)
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
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                x.append(row[0])
                y.append(float(row[1]))

            plt.plot(x, y, linestyle = 'solid',
                marker = 'o',label = cpu + "/" + backend + "/" + model + "/" + threads)
            x.clear()
            y.clear()

plt.title('Inference time for 1000 loops', fontsize = 14)
plt.legend(bbox_to_anchor=(0.80,-0.15), loc=1)
plt.tight_layout()
plt.savefig("linechart.png", format='png', dpi=1000)
plt.savefig("linechart.svg", format='svg')