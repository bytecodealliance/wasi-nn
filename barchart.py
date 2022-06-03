import os
import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('rust/examples/classification-example/build/RESULTS/testout_all-tensorflow-mobilenet_v2-2022-05-19-121720.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')

    for row in plots:
        x.append(row[0])
        y.append(float(row[1]))

plt.bar(x, y, color = 'g', width = 0.72, label = "Time ms")
plt.xlabel('Run')
plt.ylabel('Inference time in ms')
plt.title('Inference time for 1000 loops')
plt.legend()
# plt.show()
# plt.savefig(args.output + ".png", dpi=500)
plt.savefig("barchart.png", dpi=1000)