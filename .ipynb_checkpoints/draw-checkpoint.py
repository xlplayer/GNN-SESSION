import matplotlib.pyplot as plt
import os

Diginetica_P_10 = [41.23,41.17,41.28,41.29,41.29,41.28]
Diginetica_P_20 = [54.44,54.30,54.35,54.35,54.32,54.37]
Diginetica_MRR_10 = [18.34,18.34,18.40,18.35,18.38,18.37]
Diginetica_MRR_20 = [19.26,19.25,19.31,19.25,19.27,19.27]
Tmall_P_10 = [28.90,29.03,29.06,29.04,28.92,28.94]
Tmall_P_20 = [34.46,34.55,34.32,34.45,34.40,34.44]
Tmall_MRR_10 = [15.76,15.80,15.84,15.97,15.81,15.94]
Tmall_MRR_20 = [16.14,16.18,16.20,16.37,16.20,16.32]
Nowplaying_P_10 = [17.31,17.35,17.29,17.32,17.23,17.20]
Nowplaying_P_20 = [22.83,22.90,22.98,22.94,22.92,22.83]
Nowplaying_MRR_10 = [8.18,8.16,8.25,8.25,8.28,8.30]
Nowplaying_MRR_20 = [8.60,8.55,8.62,8.64,8.66,8.70]

datasets = ["Diginetica","Tmall","Nowplaying"]
metrics = ["P_10","P_20","MRR_10","MRR_20"]

x = [0,1,2,3,4,5]
fmt = ['ro-','bx-','g>-','ys-']
for dataset in datasets:
    for i,metric in enumerate(metrics):
        y = eval(dataset+"_"+metric)
        print(dataset+"_"+metric)   
        print(y)
        metric = metric.replace("_","@")
        plt.figure(0,figsize=(12,8))
        plt.title(dataset, fontsize=30)
        plt.xlabel("Maximum relative position", fontsize=30)
        plt.ylabel(metric, fontsize=30)
        plt.yticks(size = 30)
        plt.xticks(size = 30)

        plt.plot(x, y)
        plt.legend(loc='upper left',frameon = False, fontsize=30)
        if dataset is "Diginetica":
            d = dataset.lower()
        else:
            d = dataset
        plt.tight_layout()
        plt.savefig("./datasets/"+d+"/"+d+"_"+metric+".eps",format='eps',dip=600)
        print("./datasets/"+d+"/"+metric+".eps")
        plt.close(0)