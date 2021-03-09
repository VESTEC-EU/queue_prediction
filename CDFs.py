# Generates CDFs for actual_walltime/requested_walltime for a range of nodes
# This is used to generate a probability distribution of predicted walltime for a job

import pony.orm as pny
import matplotlib.pyplot as plt
import numpy as np
import datetime

from jobs import Job, initialise_database

#name of database to use
database = "Archer2.db"
#machine name
machine="ARCHER2"
#queue name to generate data from
queue = "standard_standard"


#The bins for nodesizes
nodebins = [1,2,4,8,16,32,64,128,256,512,1024]


#gets the default filename
def get_filename(machine=machine, queue=queue):
    return "%s_%s_CDF.npz"%(machine,queue)

#Object that loads a CDF file and can return randomly selected walltime ratios given:
# (optional) The number of nodes requested
# (optional) The minimum walltime (e.g. if we know a job has been running for 5h, how much longer will it run for?)
# if no options are given, will give a result based on all the jobs in the job history

# The maths theory:
# We can generate a random number from any distribution given we know its CDF
# If we have the CDF, by using a uniform random number, y, between 0 and 1,
# then a random number belonging to the distribution, x, can be calculated by:
#  x = CDF^{-1}(y) (e.g. the inverse function of the CDF)
class CDF:
    def __init__(self,file=get_filename()):
        data = np.load(file)
        self.cdf = data["cdf"]
        self.bins = data["bins"]
        self.nodebins = data["nodebins"]
        self.cdfs=data["nodedists"]

        self.n = len(self.cdf)

    def get(self,min=None,nodes=None):
        
        #select the cdf to use
        if nodes is not None:
            for n in range(len(self.nodebins)-1):
                if nodes < self.nodebins[n+1]:
                    break
            c = self.cdfs[n]
        else:
            c=self.cdf
        
        #select the minval to use
        if min is not None:
            if min > 1:
                print("Warning - minimum walltime ratio > 1. Setting to 1")
                min=1.0
            #find min index of cdf
            i = np.searchsorted(self.bins,min)
            if i>=self.n:
                #print("warning: min=%f breaks cdf"%min)
                i=self.n-1
            mn = c[i]
        else:
            mn = 0.
        
        #get a uniform random number between [min,1]
        y = mn + np.random.random()*(1-mn)
        
        #get the inverse CDF (y=CDF(x) -> x = CDF^{-1}(y))
        i = np.searchsorted(c,y)
        if i==100:
            i=99
        try:
            return self.bins[i+1]
        except Exception as e:
            print("min=",min, "nodes=",nodes,"y=",y)
            raise e








#generates the CDF of the distribution between 0 and 1
def generate_cdf(jobs):
    data, bins, patches = plt.hist(jobs, bins=100,range=(0,1),density=True,cumulative=True)

    return data,bins




if __name__ == "__main__":

    initialise_database(database)

    #get all the jobs from the database for that machine and queue, and calculate the distributions

    with pny.db_session:
        jobs = Job.select(lambda j: j.queue_name == queue and j.machine == machine)

        all=[]
        nodes=[]

        for n in range(len(nodebins)-1):
            nodes.append([])
            for j in jobs:
                if j.requested_nodes >= nodebins[n] and j.requested_nodes < nodebins[n+1]:
                    r=j.actual_walltime/j.requested_walltime
                    if r>1:
                        r=1.
                    nodes[n].append(r)

            wtimes = np.asarray(nodes[n])

            print("Range %4d-%4d: mean = %1.2f, median = %1.2f, stddev = %1.2f"%(nodebins[n],nodebins[n+1],np.mean(wtimes),np.median(wtimes),np.std(wtimes)))
        
        total = []
        for j in jobs:
            r=j.actual_walltime/j.requested_walltime
            if r>1:
                r=1.
            total.append(r)

        total = np.asarray(total)
        print("All jobs: mean = %1.2f, median = %1.2f, stddev = %1.2f"%(np.mean(total),np.median(total),np.std(total)))
        

    plt.hist(total, bins=100, density=True, range=(0,1))
    plt.ylabel("Density")
    plt.xlabel("Actual/Requested walltime")
    plt.show()

    #plot histograms
    for n in range(len(nodebins)-1):
        plt.hist(nodes[n],bins=100,density=True,alpha=0.5,label="%d-%d"%(nodebins[n],nodebins[n+1]))
    plt.xlabel("Actual/Requested walltime")
    plt.ylabel("Density")
    plt.title("")
    plt.legend()
    plt.show()

   
    print("Generating CDFs and writing %s"%get_filename())
    ntot, bins = generate_cdf(total)

    nodedists=[]
    for n in range(len(nodebins)-1):
        dist, bins = generate_cdf(nodes[n])
        nodedists.append(dist)


    

    np.savez(get_filename(),
            cdf = ntot, 
            bins=bins, 
            nodebins=nodebins,
            nodedists=nodedists
            )
        

   



