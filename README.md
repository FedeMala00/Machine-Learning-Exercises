# Machine-Learning-Exercises
Collection of the first 3 assignments of ML1 course at UniGe
## First Assignment ## 
The code first reads a data matrix from a CSV file. The data matrix is assumed to have several attribute columns and one class column.

It calculates the number of unique values for each attribute and stores this in a vector vet.

The data is then split into a training set and a test set using random permutation of the indices.

`fillCell`: Calculates likelihood probabilities for attribute values given the class.

`check_attributes`: Checks for attribute value consistency between training and test sets.

`check_dimension`: Ensures correct dimensions of training and test matrices.

`readResult`: Performs classification on the test set and calculates error rate.

Finally, it calculates and displays the error rate of the classification.

In summary, this code is implementing a Naive Bayes classifier for a dataset with categorical attributes, and it uses the classifier to predict the class labels of a test set and calculate the error rate of the predictions.

## Second Assignment ## 
The code performs the following tasks:

Data Preprocessing:

Reads the data from CSV files and performs some preprocessing.
Linear Regression - Turkish Dataset:

Performs linear regression on the Turkish dataset, containing S&P 500 and MSCI Europe index data.
Utilizes the linear_regression_turkish_dataset function.
Calculates regression coefficients and plots data points along with the regression line.
Linear Regression - Random Subsets:

Performs linear regression on different random subsets of the Turkish dataset.
Utilizes the linear_regression_random_subsets function.
Plots regression lines and data points for each subset.
Linear Regression with Intercept - 'mtcars' Dataset:

Performs linear regression with an intercept on the 'mpg' and 'weight' columns of the 'mtcars' dataset.
Utilizes the linear_regression_with_intercept_mtcars function.
Calculates regression coefficients and plots data points along with the regression line.
Multiple Linear Regression - 'mtcars' Dataset:

Performs multiple linear regression on the 'mpg', 'dis', 'hp', and 'weight' columns of the 'mtcars' dataset.
Utilizes the multiple_linear_regression_mtcars function.
Calculates regression coefficients and plots real and predicted mpg values.
Mean Squared Error (MSE) Calculation:

Calculates the Mean Squared Error (MSE) for the linear regression models on different subsets of the datasets.
Utilizes the calculate_mse function.
MSE Results Storage:

Repeats the above tasks multiple times and stores the MSE results in tables.
In summary, this code performs linear and multiple linear regression on different subsets of two datasets, calculates the MSE for each model, and visualizes the results.
## Third Assignment ## 

It loads the MNIST training and test sets.

It defines the number of classes (10, corresponding to digits 0-9) and the number of experiments to perform.

It defines a set of k values to use in the k-NN algorithm.

It initializes a results matrix and a cell array to store quality indexes.

It performs several experiments. In each experiment, it selects a random subset of the training set and performs k-NN classification for each class and each k value. It computes the accuracy of the classification and stores it in the results matrix. It also computes the confusion matrix, sensitivity, and specificity for each classification and stores them in the quality indexes cell array.

After all experiments are done, it calculates the mean and standard deviation of the sensitivity and specificity across all experiments for each class and k value.

Finally, it displays the results in tables.

In summary, this code is performing k-NN classification on the MNIST dataset for various k values and calculating various metrics to evaluate the performance of the classifier.

