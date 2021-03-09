#Generates test and train data
# Also generates random queue states based on the properties of jobs in the queue
import sys

import numpy as np
import matplotlib.pyplot as plt

import datetime
import csv

from jobs import Job, initialise_database

from CDFs import CDF

import pony.orm as pny

import datetime

#database name
dbname = "Archer2.db"

#number of nodes the system has
maxnodes = 1024

#bins for histograms
nodebins =  [1.0, 2.0, 3.0, 5.0, 9.0, 12.0, 21.0, 64.0, 910.0]
qworkbins= [0.0, 0.002777777777777778, 0.3236111111111111, 1.3488888888888888, 6.886111111111111, 24.0025, 67.1288888888889, 168.0427777777778, 9603.111111111111]
qwaitbins= [0.0002777777777777778, 1.9727777777777777, 5.151111111111111, 10.951944444444445, 21.229166666666668, 30.020555555555557, 45.53722222222222, 62.25888888888889, 485.86611111111114]
rworkbins= [0.0002777777777777778, 0.9822222222222222, 3.2447222222222223, 7.561111111111111, 14.7125, 23.1725, 47.382222222222225, 108.34722222222221, 9597.777777777777]
rremainbins= [0.0002777777777777778, 0.6266666666666667, 1.711111111111111, 3.5719444444444446, 6.221388888888889, 10.130833333333333, 14.446666666666667, 20.456666666666667, 59.275555555555556]


#queue name
queue_name = "standard_standard"



headers = [ "act_wait",
            "req_nodes",
            "req_wtime",
            "day",
            "hour",
            "s_q_jobs",
            "s_q_nodes",
            "s_q_work",
            "m_q_wait",
            "d_q_nodes0",
            "d_q_nodes1",
            "d_q_nodes2",
            "d_q_nodes3",
            "d_q_nodes4",
            "d_q_nodes5",
            "d_q_nodes6",
            "d_q_nodes7",
            "d_q_work0",
            "d_q_work1",
            "d_q_work2",
            "d_q_work3",
            "d_q_work4",
            "d_q_work5",
            "d_q_work6",
            "d_q_work7",
            "d_q_wait0",
            "d_q_wait1",
            "d_q_wait2",
            "d_q_wait3",
            "d_q_wait4",
            "d_q_wait5",
            "d_q_wait6",
            "d_q_wait7",
            "s_r_jobs",
            "s_r_nodes",
            "s_r_work",
            "d_r_nodes0",
            "d_r_nodes1",
            "d_r_nodes2",
            "d_r_nodes3",
            "d_r_nodes4",
            "d_r_nodes5",
            "d_r_nodes6",
            "d_r_nodes7",
            "d_r_work0",
            "d_r_work1",
            "d_r_work2",
            "d_r_work3",
            "d_r_work4",
            "d_r_work5",
            "d_r_work6",
            "d_r_work7",
            "d_r_remain0",
            "d_r_remain1",
            "d_r_remain2",
            "d_r_remain3",
            "d_r_remain4",
            "d_r_remain5",
            "d_r_remain6",
            "d_r_remain7"
            ]


cdf = CDF()

initialise_database(dbname)


#given a job, extracts queue information for it, returning this as a list
# If random == True, returns queue stats based on simulated walltimes of each queued job pulled from random distributions (see CDF.py)
# Otherwise uses the jobs' actual walltimes  
@pny.db_session
def get_stats(job, random=False):

    #get the job's properties
    req_nodes = job.requested_nodes
    req_walltime = job.requested_walltime/3600.
    day = job.submit_time.weekday()
    hour = job.submit_time.hour

    act_wait = job.actual_waittime/3600.

    tsubmit = job.submit_time
    tstart = job.start_time
    tstop = job.finish_time

    #get the list of queued jobs and get properties of the queue jobs etc
    queued = Job.select(lambda j: j.submit_time < tsubmit and j.start_time > tsubmit)[:]

    n_queued_jobs = len(queued)
     
    q_nodes= []
    q_work = []
    q_wait = []
    queued_work_sum = 0
    queued_nodes_sum = 0

    for q in queued:
        qnodes = q.requested_nodes
         
        if random:
            #a "fix" for jobs on cirrus belonging to Edinburgh Genomics, as they request 9999 days for a max walltime, but the jobs actually tend to only run for a few seconds 
            if "edgen" in q.queue_name:
                maxwalltime = max(3600,q.actual_walltime)/3600
            else:
                maxwalltime = q.requested_walltime/3600.
            
            qwalltime = cdf.get(nodes=qnodes) * maxwalltime
        else:
            qwalltime = q.actual_walltime/3600.

        qwork = qnodes * qwalltime
        qwait = (tsubmit - q.submit_time).total_seconds()/3600.

        q_nodes.append(qnodes)
        q_work.append(qwork)
        q_wait.append(qwait)


        queued_work_sum += qwork
        queued_nodes_sum += qnodes

        
    #get histograms of all the queued jobs' properties
    d_q_nodes, bins = np.histogram(q_nodes,bins=nodebins)
    d_q_work, bins = np.histogram(q_work,bins=qworkbins)
    d_q_wait, bins = np.histogram(q_wait, bins=qwaitbins)
    m_q_wait = 0

    if np.sum(d_q_nodes != 0):
        m_q_wait = np.median(q_wait)


    # Now do something similar for running jobs
    running = Job.select(lambda j: j.start_time < tsubmit and j.finish_time > tsubmit)[:]
    
    n_running_jobs = len(running)

    r_nodes=[]
    r_work=[]
    r_remaining=[]
    running_nodes_sum=0
    running_work_sum=0

    for r in running:
        rnodes = r.requested_nodes
        
        #the elapsed time for the running job (now - tstart)
        relapse = (tsubmit-r.start_time).total_seconds()/3600 

        #get the max walltime for the job
        if random:
            if "edgen" in r.queue_name:
                maxwalltime = max(3600,r.actual_walltime)/3600
            else:
                maxwalltime = r.requested_walltime/3600.
            
            rwalltime = cdf.get(nodes=rnodes,min=relapse/maxwalltime) * maxwalltime

        else:
            rwalltime = r.actual_walltime/3600.
        
        rremain = rwalltime-relapse
        if rremain < 0:
            print("Remaining time is negative... job that has exceeded its walltime? Setting to zero")
            #raise Exception("rwalltime is negative!")
            rremain=0 

        rwork = rnodes * rremain


        

        r_nodes.append(rnodes)
        r_work.append(rwork)
        r_remaining.append(rremain)

        running_work_sum += rwork
        running_nodes_sum += rnodes

    d_r_nodes, bins = np.histogram(r_nodes,bins=nodebins)
    d_r_work, bins = np.histogram(r_work,bins=rworkbins)
    d_r_remaining, bins = np.histogram(r_remaining, bins=rremainbins)
   
    queued_work_sum = queued_work_sum/maxnodes
    running_work_sum = running_work_sum/maxnodes

    data=generate_list([act_wait,req_nodes, req_walltime, day, hour, n_queued_jobs,queued_nodes_sum,queued_work_sum, m_q_wait, d_q_nodes, d_q_work, d_q_wait, n_running_jobs, running_nodes_sum, running_work_sum, d_r_nodes, d_r_work, d_r_remaining])


    return data

#Given a list (possibly containing sublists) returns a flattened list
# e.g. [a,b,[c,d],e] -> [a,b,c,d,e]
def generate_list(x):
    data=[]
    for item in x:
        if type(item) == type(np.zeros(1)) or type(item) == type([]):
            for y in item:
                data.append(y)
        else:
            data.append(item)
            
    
    return data





with pny.db_session:
    jobs = Job.select(lambda j: j.queue_name == queue_name)
    
    #open csv files and write their headers
    f = open("train_all.csv","w")
    g = open("test_all.csv","w")
    h = open("test_all_100.csv","w")
    
    trainwriter = csv.writer(f)
    testwriter  =csv.writer(g)
    test100writer = csv.writer(h)
    
    trainwriter.writerow(headers)
    testwriter.writerow(headers)
    test100writer.writerow(headers)
    
    
    #use test data every dday days
    dday=6
    #start date for the offset (arbitrary date)
    start = datetime.date(2020,7,1)

    i=0
    njobs = len(jobs)

    for job in jobs:
        i+=1
        day = job.submit_time.date()
        
        ## The bad way to split test and training data.
        # if i%6 == 0:
        #     data=get_stats(job,random=False)
        #     testwriter.writerow(data)
        #     print("%s: %d of %d - Test"%(day,i,njobs))
        # else:
        #     data=get_stats(job,random=False)
        #     trainwriter.writerow(data)
        #     print("%s: %d of %d - Train"%(day,i,njobs))


        
        # #This day is bad due to a rogue user submitting thousands of jobs (Cirrus) - skip it
        # if day == datetime.date(2020,8,4):
        #     print("badday")
        #     continue

        d = (day - start).days
        if d%6 != 0:
            print("%s: %d of %d - Train"%(day,i,njobs))
            data=get_stats(job,random=False)
            trainwriter.writerow(data)
        else:
           
            print("%s: %d of %d - Test"%(day,i,njobs))
            data=get_stats(job,random=False)
            testwriter.writerow(data)
            
            for j in range(100):
                data=get_stats(job,random=True)
                test100writer.writerow(data)



        
        
        

    
        




