#Takes files produced by slurm and puts the jobs within into the database

import numpy as np
import matplotlib.pyplot as plt

import datetime
import os

import glob

import pony.orm as pny

from jobs import Job,initialise_database 

#name of the database file to produce/replace
database_file = "Archer2.db"

#slurm file basename
slurm_file = "ARCHER2_Queue_"

#number of CPUs per node
CPUPerNode = 128

#Machine name
MachineName = "ARCHER2"

#machine has exclusive node access only?
exclusiveOnly=True

#column numbers in the slurm files
JOBID=0
NODES=1
CPUS=2
WALLTIME=3
SUBMIT=4
START=5
END=6
QOS=7
PARTITION=8
STATE=9

try:
    os.remove(database_file)
except FileNotFoundError:
    pass
initialise_database(database_file)

print("Deleting existing database")


#returns a datetime.datetime date from a string
def parsetime(string):
    if string == "Unknown":
        return None
    return datetime.datetime.fromisoformat(string)

#parses the walltime string from slurm into a datetime.timedelta
def parsewtime(string):
    if string == "":
        return None
    if "UNLIMITED" in string:
        return None
    if "Partition_Limit" in string:
        return None
    days=0
    if "-" in string:
        s=string.split("-")
        days = int(s[0])
        string=s[1]
    
    s=string.split(":")
    hours = int(s[0])
    minutes = int(s[1])
    seconds = int(s[2])

    return datetime.timedelta(days=days,hours=hours,minutes=minutes,seconds=seconds)



#get the list of slurm files
files = glob.glob("%s*.dat"%slurm_file)
print(files)

#sort into ascending order
files.sort()




njobs=0
exclusive=0
badwtime=0


#list of jobs that started in the previous month (to check for duplicates)
mixedmonth=[]

#loop over all files and read the jobs from them
for file in files:
    with open(file,"r") as f:
        lines=f.readlines()
    #remove header line
    lines.pop(0)
   
    count=0

    with pny.db_session:
        #loop over all jobs in the file
        # (we skip any jobs that have weird values in them - see the ifs below)
        for line in lines:
            job = line.split("|")

            #only select lines with JOBID not containing a "." (these are subjobs and we're not interested in that level of detail)
            if ("." not in job[JOBID]):
                
                if (len(job) != 11):
                    print("Weird line length",line)
                    continue
               
                id = job[JOBID]
                
                try:
                    nodes = int(job[NODES])
                except:
                    print("Funny nodes value - %s"%job[NODES])
                    continue

                #get values
                cpus = int(job[CPUS])
                requested_wtime = parsewtime(job[WALLTIME])
                tsub = parsetime(job[SUBMIT])
                tstart = parsetime(job[START])
                tstop = parsetime(job[END])
                qos = job[QOS]
                partition = job[PARTITION]
                state=job[STATE]
                
                #ignore jobs that have not completed yet
                if state == "PENDING" or state == "RUNNING":
                    continue

                #be careful with jobs who may appear in multiple files
                if tstart.month != tsub.month or tstop.month != tsub.month:
                    if id in mixedmonth:
                        #this job has already been counted
                        continue
                    mixedmonth.append(id)
                
                #some jobs seem to appear once as a node fail, and once as a completed. We don't want the duplicates
                if state == "NODE_FAIL":
                    continue

                #Don't need the extra information for cancelled jobs
                if "CANCELLED" in state:
                    state = "CANCELLED" 

                #combine partition and qos into queue name
                queue = partition+"_"+qos
                
                #number of jobs in file
                count+=1
                
                #number of jobs in total
                njobs+=1
                
                #discard jobs with unknown walltimes
                if requested_wtime is None:
                    badwtime+=1
                    continue
                
                if exclusiveOnly is False:
                    #count jobs that may not have exclusive node access
                    if cpus < CPUPerNode:
                        exclusive+=1
                        nodes = cpus/CPUPerNode

                try:
                    j=Job(uuid=id,
                        requested_nodes=nodes,
                        requested_walltime = requested_wtime.total_seconds(),
                        submit_time = tsub,
                        start_time = tstart,
                        finish_time = tstop,
                        actual_walltime = (tstop-tstart).total_seconds(),
                        machine = MachineName,
                        queue_name = queue,
                        actual_waittime = (tstart-tsub).total_seconds())
                
                except Exception as e:
                    print(line)
                    raise e



    print(file,count,len(lines))


print("Total number of jobs = %d"%njobs)
print("Percentage of jobs that may not have exclusive node access = %f %%"%(100*exclusive/njobs))
print("Percentage of jobs with unknown walltimes (and hence removed) = %f %%"%(100*badwtime/njobs))
