import numpy as np
from matplotlib.mlab import griddata



def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z


def main():
    # load the data in the form of: event, gridpoint, value
    # plot the points as a test
    # convert the plot to a contour plot



if __name__ == '__main__':
    main()