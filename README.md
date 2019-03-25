# Machine-Learning-Project-1
Probably Interesting Dataset 

# Authors
- Sandip Dey
- Shambo Ghosh

# Introduction
It is an algorithm for Expectation Maximization using Gaussian Mixture Model

# Dataset Used
- Wine Quality : https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
- Iris : https://www.kaggle.com/uciml/iris

# How to run the program
Navigate to the folder ExpMax Code and then execute the below code on terminal (must support Python 2.7 or higher)
python Wine_ExpMax.py
python Iris_ExpMax.py 

# Approach Taken

- First the data are read into a dataframe and a histogram is plotted for each feature column mentioned by the user from the list. From this we determined that optimal clusters based on likelihood that would be the best for our GMM.
- the inital parameters are chosen randomly, then the new parameters(mean, variance, and pi) are calculated via ri(the probability of the datapoint that it belongs to a cluster class)
 - Execute the Alogorithm 
            1. Execute E Step - Calculates probability of each data for each cluster
            2. Execute M Step - Calculate the weight and update mean,stat and pi
            3. Find Log likelihood
            4. If Converges Exit, else Repeat
            
# Project Structure

- Data : Datasets
- ExpMax Code : src files
- Plots - Some example graphs
            
# References
- https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php








