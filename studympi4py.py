from mpi4py import MPI
import numpy as np
from loadboomxml import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

items = None
if rank == 0:
    items = LoadResultXML('/home/hynguyen/Downloads/result_126k_0.xml')
    items = [items[:500], items[500:]]
else:
    sub_items = None

sub_items = comm.scatter(items,root=0)
print(len(sub_items))

new_data = comm.gather(sub_items, root=0)
if rank == 0:
    newnew = new_data[0] + new_data[1]
    print ('master collected: ', len(newnew))

