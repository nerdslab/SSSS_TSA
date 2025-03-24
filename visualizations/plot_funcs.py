import numpy as np
import matplotlib.pyplot as plt



def plot_input(src_x,trg_x,src_y,trg_y,dataset=None,src_id=None,trg_id=None)->tuple:
    'function for plotting sample for each class src and trg.Returns tuple of subplots (src and trgt)'
    src_x = src_x.cpu().swapaxes(1,2).numpy()
    src_y = src_y.cpu()

    trg_x = trg_x.cpu().swapaxes(1,2).numpy()
    trg_y = trg_y.cpu().numpy()

    clsses_src = list(np.unique(src_y))
    clsses_trg = list(np.unique(trg_y))

    fig_src, axs = plt.subplots(3, 2)

    ax = axs.reshape(-1)
    'plot src'
    fig_src.suptitle(f"{dataset}: {src_id}")
    #which label number of a particular label to plot.Defaults to 0! Corset maybe?
    lbl_no = 10
    for i,item in enumerate(clsses_src):
        k = np.where(src_y==item)[0]
        k = np.random.choice(k,1)[0]
        ax[i].plot(src_x[k,:])
        ax[i].set_ylim(-1.5, 1.5)
        ax[i].set_title(f"Src class {i+1}")

    plt.tight_layout()
    plt.savefig("figures/HHAR_0_2/fig_src.pdf")
    fig_trg, axs2 = plt.subplots(3, 2)
    ax2 = axs2.reshape(-1)
    'plot trg'
    for i, item in enumerate(clsses_trg):
        k = np.where(trg_y == item)[0]
        k = np.random.choice(k, 1)[0]
        ax2[i].plot(trg_x[k, :])
        ax2[i].set_title(f"Trg class {i+1}")
        ax2[i].set_ylim(-1.5,1.5)
    fig_trg.suptitle(f"{dataset}: {trg_id}")
    plt.tight_layout()
    plt.savefig("figures/HHAR_0_2/fig_trg.pdf")
    return fig_src, fig_trg


def plot_mtrx(data,x_label=None,y_label=None):
    fig, ax = plt.subplots()

    # Use 'matshow' to create a color-coded image from the matrix
    mat = ax.matshow(data,cmap='Blues')

    # Add text labels for each element (optional)
    for (i, j), z in np.ndenumerate(data):
        if z>0.5:
            c = 'white'
        else:
            c= 'black'
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize=8,color=c)


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()