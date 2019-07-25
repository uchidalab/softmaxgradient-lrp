import numpy as np
import matplotlib.pyplot as plt

def heatmap(heatmap, cmap="seismic", interpolation="none", colorbar=False, M=None):
    if M is None:
        M = np.abs(heatmap).max()
    plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, interpolation=interpolation)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar:
        plt.colorbar()