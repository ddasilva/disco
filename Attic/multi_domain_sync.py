

from mpi4py import MPI
import numpy as np

STEPS_PER_RESYNC = 50
DOMAIN_DECOMPOSITION = {
    # xmin, xmax, ymin, ymax
    0: [-np.inf,      0,       0, np.inf]            # top left
    1: [      0, np.inf,       0, np.inf],           # top right
    2: [-np.inf,      0, -np.inf,      0]            # bottom left
    3: [      0, np.inf, -np.inf,      0]            # bottom right
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

xmin, xmax, ymin, ymax = DOMAIN_DECOMPOSITION[rank]
