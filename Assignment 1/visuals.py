import matplotlib.pyplot as plt

def plot_heatmap(arr, title="Heatmap"):
    plt.figure(figsize=(6,5))
    plt.imshow(arr, origin="lower", aspect="auto")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.show()