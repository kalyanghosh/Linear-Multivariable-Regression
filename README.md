# Linear-Multivariable-Regression
To establish a relation between the dependent variable Y and a 5-tuple of independent variables X1, X2, X3, X4 and X5



## DESCRIPTION:
<br>The objective of this project is to develop a linear multivariable regression to establish a relation
between the dependent variable Y and a 5-tuple of independent variables X1, X2, X3, X4 and X5.</br>

1. To develop a pipeline of how a Data Science project is approached.

         

## DATASET:
<br>The dataset used in this project has 5 tuple feature set(X1,X2,X3,X4,X5) which are the independent variables and the dependent variable (Y) is the output variable </br>

## TASK 1. Basic Statistical Analysis:

1. For each variable Xi, i.e., column in the data set corresponding to Xi, calculate the following:
   
   histogram, mean, variance
   
   ![Histogram1](https://github.com/kalyanghosh/Linear-Multivariable-Regression/blob/master/plots/histogramX1.png)</br>
   
   
   <b>Mean of X1</b>=  40.371115600000024
   
   Correlation Matrix for all variables:
   ***************************************
          0         1         2         3         4         5
0  1.000000 -0.054110  0.058261 -0.055625  0.067539  0.956172
1 -0.054110  1.000000  0.041512  0.056898 -0.001131 -0.012087
2  0.058261  0.041512  1.000000 -0.020289 -0.053005  0.085689
3 -0.055625  0.056898 -0.020289  1.000000 -0.048236 -0.005024
4  0.067539 -0.001131 -0.053005 -0.048236  1.000000  0.133062
5  0.956172 -0.012087  0.085689 -0.005024  0.133062  1.000000
   
   
2. Use a box plot or any other function to remove outliers.

3. Calculate the correlation matrix for all variables, i.e., Y, X1, X2, X3, X4 and X5. Draw conclusions related to possible dependencies among these variables.



