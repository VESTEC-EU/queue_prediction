import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neighbors
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import time



#all the available features (columns in the csv)
cols = ["req_nodes","req_wtime","day","hour","s_q_jobs","s_q_nodes","s_q_work", "m_q_wait","d_q_nodes0","d_q_nodes1","d_q_nodes2","d_q_nodes3","d_q_nodes4","d_q_nodes5","d_q_nodes6","d_q_nodes7","d_q_work0","d_q_work1","d_q_work2","d_q_work3","d_q_work4","d_q_work5","d_q_work6","d_q_work7","d_q_wait0","d_q_wait1","d_q_wait2","d_q_wait3","d_q_wait4","d_q_wait5","d_q_wait6","d_q_wait7","s_r_jobs","s_r_nodes","s_r_work","d_r_nodes0","d_r_nodes1","d_r_nodes2","d_r_nodes3","d_r_nodes4","d_r_nodes5","d_r_nodes6","d_r_nodes7","d_r_work0","d_r_work1","d_r_work2","d_r_work3","d_r_work4","d_r_work5","d_r_work6","d_r_work7","d_r_remain0","d_r_remain1","d_r_remain2","d_r_remain3","d_r_remain4","d_r_remain5","d_r_remain6","d_r_remain7"]



#The KNN object
class NN():
    def __init__(self,weights="distance",n_neighbours = 10):
        self.nn = sklearn.neighbors.KNeighborsRegressor(weights=weights,n_neighbors=n_neighbours,metric="minkowski")
    def train(self,x,y):
        self.nn.fit(x,y)
    
    def predict(self,x):
        return self.nn.predict(x)

nn=NN()

#Takes a pandas dataframe and turns it into a matrix for use with the KNN regressor.
# also returns the vector of wait times
def to_matrix(df):
    #print("to matrix")
    nrows = len(df)
    ncols = len(cols)

    X = np.zeros((nrows,ncols))

    for col in range(ncols):
        
        c=df[cols[col]]
       
        X[:,col] = df[cols[col]]

    y = np.asarray(df["act_wait"])

    return X, y

#returns the correlation coeficient between two vectors
def correlation(x,y):
    xbar=x.mean()
    ybar = y.mean()

    top = np.sum((x-xbar)*(y-ybar))
    bottom = (np.sum((x-xbar)**2) * np.sum((y-ybar)**2))

    return np.abs(top/np.sqrt(bottom))



#applies weights to the matrix M
def apply_weights(M,weights):
    result = np.zeros_like(M)
    for i in range(len(weights)):
        result[:,i] = M[:,i]/weights[i]
    return result



#returns the root mean square of two vectors
def rms(x0,x1):
    return np.sqrt(np.mean((x0-x1)*(x0-x1)))



#normalises each column in a pandas dataframe
def normalise_df(df, means=None, stds=None):
    if means is None:
        means=[]
        stds=[]
        for col in cols:
            mean = np.mean(df[col])
            stddev = np.std(df[col])
            df[col] = (df[col]-mean)/stddev
            means.append(mean)
            stds.append(stddev)
           
        return means, stds
    else:
        i=0
        for col in cols:
            if col == "Requested Nodes":
                df[col] = np.log(df[col])
            mean=means[i]
            stddev = stds[i]
            df[col] = (df[col]-mean)/stddev
            i+=1
           






#read in test and train data
train = pd.read_csv("train_all.csv_good")
test = pd.read_csv("test_all_100.csv")
testactual = pd.read_csv("test_all.csv_good")



#normalise
means, stds = normalise_df(train)
normalise_df(testactual,means,stds)
normalise_df(test,means,stds)



#convert to matrices
trainX, trainY = to_matrix(train)
testX, testY = to_matrix(test)
testactualX, testactualY = to_matrix(testactual)


#calculate and apply weights
weights=[]
for col in cols:
    r,p = spearmanr(train[col],train["act_wait"])
    corr = correlation(train[col],train["act_wait"])
    print("%s: correlation coefficient = %f, Spearman Coefficient = %f"%(col,corr, r))
    weights.append(np.abs(r))

trainX = apply_weights(trainX,weights)
testactualX = apply_weights(testactualX,weights)
testX = apply_weights(testX,weights)



print("Training")
nn.train(trainX,trainY)


start=time.time()

print("Predicting using actual queue stats")
yy=nn.predict(testactualX)

stop=time.time()
print("Took %f seconds"%(stop-start))

print("Error = %f"%rms(testactualY,yy))

n=len(yy)
err = np.abs(yy-testactualY)
err.sort()
for i in range(n):
    if np.abs(err[i]) > 1.:
        print("Prediction is correct within 1 hour = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 2.:
        print("Prediction is correct within 2 hours = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 6.:
        print("Prediction is correct within 6 hours = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 12.:
        print("Prediction is correct within 12 hours = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 24.:
        print("Prediction is correct within 24 hours = %f%%"%(i/n *100))
        break



plt.plot(testactualY,yy,".")
plt.plot([0,np.max(testactualY)],[0,np.max(testactualY)])
plt.xlabel("Actual wait time (h)")
plt.ylabel("Predicted wait time (h)")
plt.show()

print("")

print("Predicting using randomly genrated queue stats (may take a few mins)")

start=time.time()
yy=nn.predict(testX)
stop=time.time()
print("Took %f seconds"%(stop-start))



nsamples=100

njobs = len(yy)//nsamples

#calculate the mean and std of the predictions
ypred = []
yerr = []
for i in range(njobs):
    ys=[]
    for j in range(nsamples):
        ys.append(yy[j+i*nsamples])
    
    ypred.append(np.mean(ys))
    yerr.append(np.std(ys))

ypred = np.asarray(ypred)
yerr = np.asarray(yerr)

#save the results in case they are needed again
np.savez("results100.npz",actual=testactualY, ypred=ypred, yerr=yerr)

##restore the results
# data=np.load("results100.npz")
# ypred = data["ypred"]
# yerr = data["yerr"]

print("Error = %f"%rms(testactualY,ypred))

plt.plot(testactualY,ypred,".")
plt.plot([0,np.max(testactualY)],[0,np.max(testactualY)])
plt.xlabel("Actual wait time (h)")
plt.ylabel("Predicted wait time (h)")
plt.show()


err = (ypred-testactualY)/yerr

plt.hist(err,bins=100)
plt.show()

plt.hist(yerr,bins=100)
plt.show()

err = np.abs(err)
err.sort()
n=len(err)

for i in range(n):
    if np.abs(err[i]) > 1.:
        print("Prediction is correct within 1-sigma = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 2.:
        print("Prediction is correct within 2-sigma = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 3.:
        print("Prediction is correct within 3-sigma = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 4.:
        print("Prediction is correct within 4-sigma = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 5.:
        print("Prediction is correct within 5-sigma = %f%%"%(i/n *100))
        break

print("")
err =  np.abs(ypred-testactualY)

err.sort()
for i in range(n):
    if np.abs(err[i]) > 1.:
        print("Prediction is correct within 1 hour = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 2.:
        print("Prediction is correct within 2 hours = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 6.:
        print("Prediction is correct within 6 hours = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 12.:
        print("Prediction is correct within 12 hours = %f%%"%(i/n *100))
        break
for i in range(n):
    if np.abs(err[i]) > 24.:
        print("Prediction is correct within 24 hours = %f%%"%(i/n *100))
        break

print("")
y = testactualY
y.sort()
for i in range(n):
    if np.abs(y[i]) > 1.:
        print("Proportion of jobs with wait < 1h = %f%%"%(i/n *100))
        break













