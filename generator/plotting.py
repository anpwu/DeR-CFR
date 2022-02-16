import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import os

def draw_w(coef1,coef2,coef3,file_path):
    # size
    plt.figure(figsize=(20, 8), dpi=80)

    x = max([coef1.shape[0],coef2.shape[0],coef3.shape[0]])

    # color
    plt.plot(range(0, coef1.shape[0]), coef1, label="t_AB", color="#F08080")
    plt.plot(range(0, coef2.shape[0]), coef2, label="y1_BC", color="#0000FF", linestyle="--")
    plt.plot(range(0, coef3.shape[0]), coef3, label="y2_BC", color="#102020", linestyle="-.")

    # x axis
    _xtick_labels = range(0, x)
    plt.xticks(_xtick_labels)
    # plt.yticks(range(0,9))

    # gird
    plt.grid(alpha=0.4, linestyle=':')

    # legend
    plt.legend(loc="upper left")

    # save
    plt.savefig(file_path + 'weight' + '.png')

    # show
    plt.show()