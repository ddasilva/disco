#!/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 4

comm.send(rank, dest=(rank + 1) % n)
got = comm.recv(source=(rank - 1) % n)

print(f"Process {rank} Got {got}")
