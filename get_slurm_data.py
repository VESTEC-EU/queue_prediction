#Run on an HPC machine with SLURM

import os
import datetime
from dateutil.relativedelta import relativedelta

machine="ARCHER2"

#starty and stop dates to get data for
start = datetime.date(2020,7,1)
stop = datetime.date(2021,2,1)

#get data month by month (so as to not put too much of a strain on slurm)
d=start
while (d<stop):
    st = d+relativedelta(months=1)
    
    print(d,st)
    command = "sacct -a -p --format=JobID%%30,ReqNodes,NCPUS,TimeLimit%%20,Submit,Start,End,QOS%%20,Partition%%20,State%%20 -S %s -E %s > %s_Queue_%s_%s.dat"%(d,st,machine,d,st)
    os.system(command)

    d = st


