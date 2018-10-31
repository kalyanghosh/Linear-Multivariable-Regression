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
   
Comment: From the above mean table, we see that the mean of the variables is increasing as we go from (X1, X2, X3, X4, X5). The means of the variables also increase by roughly about (60) as we go from X1to X2, by about (53) from X2 to X3, by about (85) as we go from X3 to X4 and by about (36) as we go from X4 to X5.

Comment: From the above variance table, we see that the variables X1,X2 and X4 have about the same amount of variance and variables X3 and X5 have around the same amount of variance but it is lower than the variance of the variables (X1,X2 and X4).
   
 
   

1.2 Use a box plot or any other function to remove outliers (do not over do it!). This can also be
done during the model building phase (see tasks 2 and 3).




13. Calculate the correlation matrix for all variables, i.e., Y, X1, X2, X3, X4 and X5. Draw conclusions related to possible dependencies among these variables.

![CM](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/correlation_matrix.JPG)</br>

