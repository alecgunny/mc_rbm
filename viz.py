import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import shutil
import os
import subprocess

class WeightPlotter:
    '''
    class for plotting the evolution of model weights over the course of training
    adds some extra scaffolding to keep plotting fast
    Parameters
    --------------
    -W: keras.variable instance
        a symbolic variable representing the model weight matrix
    '''
    def __init__(self, W):
        self.W = W

        h_dim = W.get_value().shape[1]
        n_cols = int(np.round(np.sqrt(h_dim)))
        n_rows = n_cols if n_cols**2 >= h_dim else n_cols + 1

        self.fig = plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(n_rows, n_cols)
        gs.update(wspace=0.015, hspace=0.015)
        self.glyphs = []
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i*n_rows + j
                if idx >= h_dim:
                    break
                ax = self.fig.add_subplot(gs[idx])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                self.glyphs.append(ax.imshow(np.zeros((28,28))))

        output_dir = './imgs/weights/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def _save(self, epoch):
        # save the current figure and label the file by `epoch`
        epoch = str(epoch).zfill(4)
        self.fig.savefig(os.path.join(self.output_dir, 'epoch{}.png'.format(epoch)), bbox_inches='tight', pad_inches=0)

    def plot(self, epoch):
        # plot the model weights on epoch `epoch`
        W = self.W.get_value().T
        for w, glyph in zip(W, self.glyphs):
            glyph.set_data(w.reshape(28,28))
            glyph.set_clim(vmin=w.min(), vmax=w.max())
            plt.draw()
        self._save(epoch)

    def convert(self):
        # convert the plotted images to a gif using ffmpeg. Converts to .avi first
        ffmpeg_args = ['-nostats', '-loglevel', 'panic', '-hide_banner', '-y']
        subprocess.call(['ffmpeg', '-i', './imgs/weights/epoch%04d.png', './imgs/weights.avi'] + ffmpeg_args)
        subprocess.call(['ffmpeg', '-i', './imgs/weights.avi', '-pix_fmt', 'rgb8', '-t', '3', './imgs/weights.gif'] + ffmpeg_args)


def plot_sample(x, pred):
    fig = plt.figure()
    ax_left = fig.add_subplot(121)
    ax_left.xaxis.set_visible(False)
    ax_left.yaxis.set_visible(False)

    ax_rite = fig.add_subplot(122)
    ax_rite.xaxis.set_visible(False)
    ax_rite.yaxis.set_visible(False)

    ax_left.imshow(x.reshape(28, 28))
    ax_left.set_title('Original')

    ax_rite.imshow(pred.reshape(28, 28))
    ax_rite.set_title('Reconstruction')

    plt.tight_layout()
    fig.show(0)

    return fig
