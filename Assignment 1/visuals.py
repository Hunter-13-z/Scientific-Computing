import matplotlib.pyplot as plt

def plot_heatmap(arr, title="Heatmap", show=True):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    im = ax.imshow(
        arr,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="plasma",   # same colormap
        vmin=0,          # same min
        vmax=1           # same max
    )

    fig.colorbar(im, ax=ax, label="Concentration")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)