import matplotlib.pyplot as plt
import numpy as np

def plot_attention_matrix(attn, filename="attention.png"):
    """
    attn: torch.Tensor ou np.array de shape [T, T]
    """
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()

    T = attn.shape[0]

    fig, ax = plt.subplots(figsize=(T, T))
    ax.set_xlim(0, T)
    ax.set_ylim(0, T)
    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(T))
    ax.set_xticklabels(range(T))
    ax.set_yticklabels(range(T))
    ax.grid(True)

    # inverser l'axe Y pour que (0,0) soit en haut à gauche
    ax.invert_yaxis()

    max_val = attn.max()

    for i in range(T):
        for j in range(T):
            value = attn[i, j]

            if value <= 0:
                continue

            radius = 0.45 * (value / max_val)

            circle = plt.Circle(
                (j + 0.5, i + 0.5),
                radius,
                color="black",
                fill=False,
                linewidth=1.5
            )
            ax.add_patch(circle)

            ax.text(
                j + 0.5,
                i + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title("Attention Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
