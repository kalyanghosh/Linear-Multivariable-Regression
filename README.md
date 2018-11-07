# Linear-Multivariable-Regression
To establish a relation between the dependent variable Y and a 5-tuple of independent variables X1, X2, X3, X4 and X5



## DESCRIPTION:
<br>The objective of this project is to develop a linear multivariable regression to establish a relation
between the dependent variable Y and a 5-tuple of independent variables X1, X2, X3, X4 and X5.</br>

1. To develop a pipeline of how a Data Science project is approached.

         

## DATASET:
<br>The dataset used in this project has 5 tuple feature set(X1,X2,X3,X4,X5) which are the independent variables and the dependent variable (Y) is the output variable </br>

## TASK 1. Basic Statistical Analysis:

1.1 For each variable Xi, i.e., column in the data set corresponding to Xi, calculate the following:
   
   histogram, mean, variance
   
   ![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/histogram_of_variables1_1.png)</br>
   
   ![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/means.JPG)</br>
   
   ![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/variance.JPG)</br>
   
<b>Comment:</b> From the above mean table, we see that the mean of the variables is increasing as we go from (X1, X2, X3, X4, X5). The means of the variables also increase by roughly about (60) as we go from X1to X2, by about (53) from X2 to X3, by about (85) as we go from X3 to X4 and by about (36) as we go from X4 to X5.

<b>Comment:</b> From the above variance table, we see that the variables X1,X2 and X4 have about the same amount of variance and variables X3 and X5 have around the same amount of variance but it is lower than the variance of the variables (X1,X2 and X4).
   
 
   

1.2 Use a box plot or any other function to remove outliers (do not over do it!). This can also be
done during the model building phase (see tasks 2 and 3).


 ![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/boxplot1.JPG)</br>
 
 
<b>Comment:</b> From the above box plots, we see that median of the values for the variable X1 is around 40, the median for X2 is around 100, median for X3 is around 150, median for X4 is around 230 and for X5 is around 265.  
We also see from the box plot of Y, that the median value lies around 6100.


Interquartile Ranges:

![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/iqr1.JPG)</br>

We now try to remove the outliers in the data by rejecting data points which are beyond the range Q1-1.5*IQR and Q3+1.5*IQR

![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/boxplot2.JPG)</br>

![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/boxplot3.JPG)</br>


We see, that most of the outliers are removed after outlier removal.



1.3 Calculate the correlation matrix for all variables, i.e., Y, X1, X2, X3, X4 and X5. Draw conclusions related to possible dependencies among these variables.

![CM](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/cm1.JPG)</br>

<b>Comment:</b> In the above correlation matrix, columns and rows (0 to 4) correspond to the variables X1 to X5 and the column and row (5) correspond to the o/p variable Y.
In the above correlation matrix, we see that the dependent variable Y has strong relation with the input variable X1 which is indicated by the coefficient (0.962).
We also see, that apart from variable X1, there is not much strong relation between the dependent variable and the in the independent variables.
We now, look into the correlations among the independent variables. We see that there is negative correlation between X1 and X2, X4 and positive correlation between X1 and X3, X5 however this correlation is small.
Similarly, there are small positive & negative correlations among the different independent variables, but those correlations are not of much significance as the values are all less than 0.1


## TASK 2: Simple Linear Regression:
Carry out a simple linear regression to estimate the parameters of the model: Y = a0 + a1X1 + ε.

2.1 Determine the estimates for a0, a1, and σ2.
Value of a0= 3226.11
Value of a1= 83.14
Value of variance= 205627.37

Comment:

The value of a0 is the expected value of Y when our variable is 0
The value of a1 is the coefficient for the variable X1. It is also the slope of the straight line that we are fitting.
The variance is the is the error term which accounts for the randomness in our data which our model cannot explain.

2.2 Check the p-values, R2, and F value to determine if the regression coefficients are significant.

![CM](![CM](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/cm1.JPG)</br>

