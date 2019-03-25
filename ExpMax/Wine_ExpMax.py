
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random as rnd

class gmmExpMax:
    def __init__(self,X,iterations):
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.var = None

    def expmax(self):

        # initiate initial random guesses for mean, variance, and probability
        self.mu = [rnd.uniform(self.X.min(), self.X.max()), rnd.uniform(self.X.min(), self.X.max())]
        self.pi = [1/2.0,1/2.0]
        self.var = [rnd.uniform(0,2), rnd.uniform(0,2)]
                
        # Expectation step
        for iter in range(self.iterations):
            #Create array with dimensionality of data
            r = np.zeros((len(self.X),2))

            # Probability datapoint is in gaussian model
            for c,g,p in zip(range(2),[norm(loc=self.mu[0],scale=self.var[0]),
                                       norm(loc=self.mu[1],scale=self.var[1])],self.pi):
                r[:,c] = p*g.pdf(self.X)
            
            # Normalize probabilities to sum to 1
            for i in range(len(r)):
                r[i] = r[i]/np.sum(self.pi)*np.sum(r,axis=1)[i]

            # Maximization step
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:,c])
                m_c.append(m)

            # calculate probability for each cluster
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k]/np.sum(m_c))
            self.mu = np.sum(self.X.reshape(len(self.X),1)*r,axis=0)/m_c
            var_c = []
            for c in range(len(r[0])):
                var_c.append((1/m_c[c])*np.dot(((np.array(r[:,c]).reshape(1599,1))*(self.X.reshape(len(self.X),1)-self.mu[c])).T,(self.X.reshape(len(self.X),1)-self.mu[c])))

            self.var = np.squeeze(var_c)

        return self.mu, self.var

def initialcluster(datanp):
    optCluster = 0
    for cluster in range(2,10):
        logs=[]
        #checking cluster feasibility   
        data= np.empty(len(datanp))
        np.copyto(data,datanp)
        stat = Stat()
        stat.SetInitialValues(cluster,data)
        for x in range(10):
            rValue = EStep(data,cluster,stat)
            stat = MStep(data,cluster,stat,rValue)
            logLikely = likelyhood(data,stat)
            logs.append(logLikely)
        if all(x<y for x, y in zip(logs, logs[1:])):
            optCluster = cluster
            break
    return optCluster


def likelyhood(data,Stat: Stat):
    logLik = 0
    for x in range(len(Stat.mean)):
        varS = Stat.variance[x] * Stat.variance[x]
        term1 = ((len(data)) * math.log(2 * math.pi))/(-1*2)
        term2 = ((len(data)) * math.log(varS))/(-1*2)
        term3 = np.sum((np.power(np.subtract(data,Stat.mean[x]),2))/(-1*2*varS),axis=0)
        logLik += term1+term2+term3
    return logLik


def main():
    data = pd.read_csv('data/winequality-red.csv')
    data.replace('', np.nan, inplace=True)
    data = data.dropna()

    rnd.seed(1)
    flag = True

    while flag:
        #Get Numeric Columns from Dataset
        numCols = list(data.select_dtypes([np.number]).columns.values)

        print('Available Columns')
        for i in range(len(numCols)):
            print(str(i) + "  " + numCols[i])

        colNo = input("Select one Column(Enter number): ")

        #Retrieve Column data based on user input
        dataArr = data.iloc[:,data.columns.get_loc(numCols[int(colNo)])].values

        myfeature = data[[data.columns[colNo]]].copy()

        print('Close the Graph to continue.')

        #Shows the Histogram of Data
        plt.hist(dataArr, bins='auto')
        plt.title("Histogram for Column : " + numCols[int(colNo)])
        plt.plot()
        plt.show()

        # Fit Gaussian Mixture model to feature
        clusters = initialcluster(dataArr) #2 
        iterations = 2
        x = np.array(dataArr)
        xrav = x.ravel()
        gmm = gmmExpMax(xrav, iterations)
        mu, sigma = gmm.expmax()

        x = np.linspace(x.min(), x.max(), 1599, endpoint=False)
        ycomb = np.zeros_like(x)


        # Create plot for GMM
        for s in range(0, clusters):
            y = norm.pdf(x, mu[s], sigma[s])
            plt.plot(x,np.transpose(y))
            ycomb = ycomb + np.transpose(y)

        plt.plot(x,ycomb)

        print('Close the Graph to continue.')
        # Set up labels and show plot
        plt.xlabel(myfeature.columns[0])
        plt.ylabel('percentage')
        plt.show()

        choice = str(input('Press Y to Restart and N to Exit: '))

        if str.lower(choice) == 'n':
            flag = False


if __name__ == "__main__":
    main()           

