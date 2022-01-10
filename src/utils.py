import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# We normalize the images to be able to better use heatmaps and visualize residuals.
def plot_recon_wall(xhat, yhat ,cmap, seed=33, savefig=False, savename='Wall_recon_plot.png',
                    ncols=5, nrows=3, norm=True):
    """
    Plots a customized wall of images with N-rows and N-columns (3 x 5 by default). Function assumes first image is
    the original image, second is a reconstructed image and by default the last one is the residual of the 2.
        Parameters
        ----------
        xhat     : tensor
            original images.
        yhat      : tensor
            ML recreated image.
        cmap     : matplotlib.cm object
            colormap to use when displaying images. By default, the residual image (Original - ML)
            uses seismic colormap.
        seed     : int
            sets the seet to passed value.
        savefig  : bool
            saves the image to current directory when set to True.
        savename : str
            when `savefig` is set to true, you can add a custom name to save the image.
        ncols    : int
            number of columns to use in the plotting wall. Set to 5 by default.
        nrows    : int
            number of rows to use in the plotting wall. Set to 3 by default. First row
            is the original image by default, second is the recreated image (MLR) and the
            last row is the residual (original - ML).
        norm     : bool
            set to true to normalize images [0, 1].
    """
    plt.close('all')
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5.6))

    np.random.seed(seed=seed)
    nums = np.random.randint(0,len(xhat), ncols)

    normalize = lambda z : z / (z.max() - z.min()) if norm else z

    first_row_imgs = [] # Original MNIST image
    second_row_imgs = [] # ML Reconstructed image
    third_row_imgs = [] # Original - MLR


    for i in range(ncols):
        xhat_norm = normalize(xhat[nums[i], 0, :, :]) # Original normalized img
        yhat_norm = normalize(yhat[nums[i], 0, :, :]) # ML normalized img


        Original = axis[0, i].imshow(xhat_norm, interpolation='bilinear',
                               cmap=cmap, aspect='equal')

        ML = axis[1, i].imshow(yhat_norm, interpolation='bilinear',
                               cmap=cmap, aspect='equal')

        res = xhat_norm - yhat_norm # Original - ML

        residual = axis[2, i].imshow(res, vmin=-1, vmax=1, interpolation='bilinear',
                          cmap=cm.seismic, aspect='equal')



        first_row_imgs.append(Original)
        second_row_imgs.append(ML)
        third_row_imgs.append(residual)


    for ax in axis.ravel():
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)


    # Row labels
    axis[0,0].text(0.5, 2, s='MNIST image', bbox={'facecolor': 'white', 'pad': 2.0})
    axis[1,0].text(0.5, 2, s='ML Recreated Img', bbox={'facecolor': 'white', 'pad': 2.0})
    axis[2,0].text(0, 2, s='Residual', bbox={'facecolor': 'white', 'pad': 2.0})

    fig.subplots_adjust(wspace=0, hspace=0) #spacing between plots

    #adding color bar for residuals
    cb_ax = fig.add_axes([0.7,0.099,0.20,.03]) # param - rect : This parameter is the dimensions [left, bottom, width, height] of the new axes.
    ax_list = [axis[-1,i] for i in range(ncols)]
    cbar = plt.colorbar(third_row_imgs[0], ax = ax_list, fraction = 0.019, pad = 0.02, cax=cb_ax, ticks=[-1,0,1], orientation="horizontal")
    cbar.ax.get_xaxis().labelpad = 15
    cbar.ax.set_xlabel('Residual')

    if savefig:
        plt.savefig(savename, dpi=300, bbox_inches='tight')
    plt.show();
