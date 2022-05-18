
## Multivariate Regression
In multivariate situation, we have more than one input variables that will contribute information to the prediction of dependent variable. Suppose we have p features as inputs, the independent variables can be denoted as $X_1, X_2, X_3...X_p$

Thus the relationship between y and X, take the model $F$, we have:

$E(y|X_1,X_2...X_p)=F(X_1,X_2...X_p)$

In linear regression, we have $p+1$ numbers of beta coefficients with $\beta_0$ represent the intercept. So the functional relationship between X and y is:
 
 $E(y|X_1,X_2...X_p)=\beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p$

If we let $X$ be the matrix of dependent variables. Suppose we have i observations with p numbers of features, $X$ is a $i\times p$ matrix. Let $\beta$ be a vector of coefficients. Then we have:

$E(y) = \beta X$

We can then solve for $\beta$ by multiplying $X^T$ to both sides:

$X^Ty = \beta X^TX+ \epsilon X^T$

$X^Ty = \beta X^TX+ 0 X^T$

$\hat{\beta} = (X^tX)^{-1}X^t Y$

For locally weighted regression, we will add weighting terms to the computation of $\beta$:

$\hat{\beta} = (X^{T}WX)^{-1}X^{T}WY$

However, when the number of features rise to a large value, sometime not all variables contribute information to the prediction of y. Then we will need a variable selection algorithm to pick out features that are not important to the model. We will assign weights $w_i$ to each of the features. 0 represents an unimportant feature that we want to excluded from our model and 1 represent an important one. Common variable selection approaches include regularization methods such as Ridge, Lasso and ElasticNets. We can use these algorithm to reconstruct the sparsity pattern of $W$


## Gradient Boosting
After we use our initial regression model to estimate the dependent variable with input features, we will have residuals. 

$\text{residuals}_i= \beta_ix_i-y_i$ 

To improve the accuracy of predictions by the initial model (we considered as a weak learner), scientists introduced a method to train another model on the error terms with respect to x. This is gradient boosting. The most common approach is to train a decision tree as the boosting algorithm, but there are many choices of boosters. The final prediction after gradient booster will be the sum of predictions made by the weak learner and prediction made by boosting algorithm. 



## XGBoost
XGBoost, or Extreme gradient boosting, uses regularization parameters to avoid overfitting. Same as gradient boosting discussed above, XGBoost can use deicision tree based boosting method.

  

After we fit the original model, we calculate the residuals for each data point. We then implement decision tree to the residuals. The methods for splits is maximizing __Gain__, which is the improvement in accuracy. There are two important hyperparameter in the Gain function to control how we split the tree: $\lambda$, and $\gamma$. $\lambda$ is used to avoids overly sensitivity to individual data points, and $\gamma$ is the threshold to stop further splitting the tree.

  

The function for Gain is applied with each split:

 $Gain = \frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}- \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma$

G_L is the sum of residuals in the left node, G_R is the sum of residuals in the right node, H_R is the number of residuals in the left node and H_L is the number of residuals in the right node.

  

For each node, we use the Gain function to decide where is the best place to split the tree. The node wil be splitted at where the Gain function achieve the highest value. The algorithm will examine each possible way of splitting the tree. The tree will stop further developing when the Gain function outputs negative values. Each node will be examined and splitted until all ways of splitting will yield negative values for splitting.

  

After the first decision tree is constructed based on the residuals, we have a new model. The prediction now equals to the initial prediction plus the prediction made by the decision tree learner multiplies a learning rate. The new model will have a new residuals when it is compared to the real values of y. The XGBoost algorithm will build another decision tree based on the new residuals. This process is repeated n times(a designated values). The final prediction will be as follows

 $$Prediction = Initial \text{ }Prediction + Learning Rate \times Prediction_1 + Learning Rate \times Prediction_2 + ....Learning Rate \times Prediction_n$$

In this project, I will show the steps in writing a gradient boosting algorithm with a random forest regressor as booster. Then I will applied the gradient boosted regression algorithm to the analysis of a real dataset and compare it with prediction made by the original regression algorithm and xgboost. The performance of these models will be evaluated by a cross-validated mean square error. 

I will the Boston housing dataset. Features will be used to predict cmedv, the mean price of the house. I will first use a locally weighted regression model to fit the data. Then, I will use a random forest model to boost the locally weighted regression model. 

```
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

#Define the locally weighted regression model
def lwr_reg(X, y, xnew, kern, tau, intercept):

    n = len(X)
    yest = np.zeros(n)

    if len(y.shape)==1:
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) 

    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        theta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],theta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew)
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output


#Define the kernel we use for the locally weighted regression
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2))


def CrossMSE(X,y,kernel,tau,intercept):
  mse_i = []
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  for idxtrain, idxtest in kf.split(X):
      xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      ytest = y[idxtest]
      xtest = X[idxtest]
      xtrain = scale.fit_transform(xtrain)
      xtest = scale.transform(xtest)
      yhat = lwr_reg(xtrain,ytrain,xtest,kernel,tau,intercept)
      mse_i.append(mse(ytest,yhat))
  return np.mean(mse_i)


# Import the data and assign x feature and y values
data = pd.read_csv('/content/drive/MyDrive/Data410/Non-linearReg/BostonHousingPrices.csv')
x = data[['crime', 'ptratio', 'rooms', 'older','lstat']].values
y = data['cmedv'].values


scale = StandardScaler()
x_scaled = scale.fit_transform(x)

mse_lwr = CrossMSE(x_scaled,y,Tricubic,0.5,True)
print('The cross-validated mean square error of locally weighted regression is '+str(mse_lwr))
```
The cross-validated mean square error of locally weighted regression is 17.817376346186034

```
#here is the function of gradient boosted lowess model
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  y_initial= lwr_reg(X,y,X,kern,tau,intercept) 
  new_y = y - y_initial # we compute the residulas
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  model.fit(X,new_y) #fit the random forest regressor to x and residuals
  output = model.predict(xnew) + lwr_reg(X,y,xnew,kern,tau,intercept) #the output equals to the sum of prediction made by the original model and predictions made by the booster
  return output

def CrossMSE_Boosted(X,y,kernel,tau,intercept):
  mse_i = []
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  for idxtrain, idxtest in kf.split(X):
      xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      ytest = y[idxtest]
      xtest = X[idxtest]
      xtrain = scale.fit_transform(xtrain)
      xtest = scale.transform(xtest)
      yhat = boosted_lwr(xtrain,ytrain,xtest,kernel,tau,intercept)
      mse_i.append(mse(ytest,yhat))
  return np.mean(mse_i)


mse_gboost = CrossMSE_Boosted(x,y,Tricubic,0.5,True)

print('The cross-validated mean square error of locally weighted regression with gradient boosting is '+str(mse_gboost))
```
The cross-validated mean square error of locally weighted regression with gradient boosting is 17.728742021954826
```
import xgboost as xgb

model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)

def CrossMSE_xgb(X,y):
  mse_i = []
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  for idxtrain, idxtest in kf.split(X):
      xtrain = X[idxtrain]
      ytrain = y[idxtrain]
      ytest = y[idxtest]
      xtest = X[idxtest]
      xtrain = scale.fit_transform(xtrain)
      xtest = scale.transform(xtest)
      model_xgb.fit(xtrain,ytrain)
      yhat = model_xgb.predict(xtest)
      mse_i.append(mse(ytest,yhat))
  return np.mean(mse_i)

mse_xgb=CrossMSE_xgb(x,y)
print('The cross-validated mean square error of xgboost regressor is '+str(mse_xgb))
```
The cross-validated mean square error of xgboost regressor is 15.274251246203415

The result show that xgboost outperformed both locally weighted regression and gradient boosted locally weighted regression. The mean square error reported by xgboost is the lower among all three regressors. The implementation of Gradient boost increase the accuracy of locally weighted regression. But it is not as efficient as xgboost. Since xgboost has multiple boosters, this result is much expected. 


Reference:
https://towardsdatascience.com/xgboost-python-example-42777d01001e

