import numpy as np
import matplotlib.pyplot as plt

def show_heatmap(heatmap, relu=False, cmap="seismic", colorbar=False, M=None):
    if relu:
        heatmap = heatmap * (heatmap > 0)

    if M is None:
        M = np.abs(heatmap).max()
    plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar:
        plt.colorbar()