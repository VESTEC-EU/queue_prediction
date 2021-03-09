#Suggests good histogram bin edges for queue features

import numpy as np
import matplotlib.pyplot as plt

import pony.orm as pny

from jobs import Job, initialise_database

import datetime


initialise_database("Archer2.db")


#Given a numpy array (arr), will generate bin boundaries for nbins bins such that each bin has approximately the same number of elements in it
def get_bins(arr,nbins=8):
    arr = np.sort(arr)

    n = len(arr)

    dn = n//nbins
    
    bins=[]
    bins.append(np.min(arr))
   
    for i in range(nbins-1):
        old = arr[dn]
        while (True):
            if arr[dn+1] != old and arr[dn+1] != arr[0]:
                #print("success", arr[0], arr[dn+1])
                dn+=1
                break
            dn +=1
            old = arr[dn]
            if dn > n:
                print("Whoops")
                print(len(bins))
                print(bins)
                raise Exception("Bad")
        bins.append(arr[dn])
        # print(i, dn, arr[dn], n)
        dn = dn + (n-dn)//(nbins-(i+1))
        
    bins.append(arr[-1])
    # print(len(bins))
    # print(bins)
    # print(np.max(arr))
    return bins




with pny.db_session:
    jobs = Job.select(lambda j: j.queue_name == "standard_standard")

    nodes=[]
    qwork=[]
    qwait=[]
    rwork=[]
    rremain=[]

    i=0
    for job in jobs:


        i+=1
        if i%10 !=0:
            continue

        print("%d of %d"%(i,len(jobs)))


        nodes.append(job.requested_nodes)
        
        tsub = job.submit_time

        queued = Job.select(lambda j: j.submit_time < tsub and j.start_time > tsub)
        for q in queued:
            qwait.append((tsub-q.submit_time).total_seconds()/3600.)
            qwork.append(q.requested_nodes * q.actual_walltime / 3600.)
        
        running = Job.select(lambda j: j.start_time < tsub and j.finish_time > tsub)
        for r in running:
            tremain = (r.finish_time-tsub).total_seconds()/3600
            rwork.append(tremain*r.requested_nodes)
            rremain.append(tremain)

    nodes = np.asarray(nodes)
    qwork = np.asarray(qwork)
    qwait = np.asarray(qwait)
    rwork = np.asarray(rwork)
    rremain = np.asarray(rremain)


    nodebins = get_bins(nodes)
    print("nodebins = ", nodebins)
    plt.hist(nodes,bins=nodebins,alpha=0.5)
    plt.hist(nodes,alpha=0.5)
    plt.show()

    qworkbins = get_bins(qwork)
    print("qworkbins=",qworkbins)
    plt.hist(qwork,bins=qworkbins,alpha=0.5)
    plt.hist(qwork,alpha=0.5)
    plt.show()

    qwaitbins = get_bins(qwait)
    print("qwaitbins=",qwaitbins)
    plt.hist(qwait,bins=qwaitbins,alpha=0.5)
    plt.hist(qwait,alpha=0.5)
    plt.show()

    rworkbins = get_bins(rwork)
    print("rworkbins=",rworkbins)
    plt.hist(rwork,bins=rworkbins,alpha=0.5)
    plt.hist(rwork,alpha=0.5)
    plt.show()

    rremainbins = get_bins(rremain)
    print("rremainbins=",rremainbins)
    plt.hist(rremain,bins=rremainbins,alpha=0.5)
    plt.hist(rremain,alpha=0.5)
    plt.show()









