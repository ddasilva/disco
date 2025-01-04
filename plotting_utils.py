from matplotlib.patches import Wedge 
from matplotlib import pyplot as plt
import numpy as np

def add_earth(ax=None):
    """Add an earth with dayside/nightside shaded to plot.
    
    Arguments
      ax: axis to draw onto
    """
    if ax is None:
        ax = plt.gca()

    theta1, theta2 = 180, 270
    radius = 1
    center = (0, 0)
    
    theta1, theta2 = 90, 270
    radius = 1.0
    center = (0, 0)
    w1 = Wedge(center, radius, theta1, theta2,
               fc='#333333', edgecolor='black', zorder=999)
    w2 = Wedge(center, radius, theta2, theta1,
               fc='white', edgecolor='black', zorder=999)
    
    for wedge in [w1, w2]:
       ax.add_artist(wedge)
