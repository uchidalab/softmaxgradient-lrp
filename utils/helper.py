import numpy as np
import matplotlib.pyplot as plt

def heatmap(heatmap, relu=False, cmap="seismic", interpolation="none", colorbar=False, M=None):
    if relu:
        heatmap = heatmap * (heatmap > 0)

    if M is None:
        M = np.abs(heatmap).max()
    plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, interpolation=interpolation)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar:
        plt.colorbar()