---
layout: post
title: Regularization in Linear Models
description: Ridge and Lasso Regression
date: 2018-09-01 08:00:00
image: /assets/images/ridge_lasso.png
tags:
    - python
    - machine-learning
    - regularization
comments: true
bokeh: true
---

Least squares estimates are often not very satisfactory due to their poor out-of-sample performance, especially when the model is overly complex with a lot of features. We can attribute this to low bias and large variance in least squares estimates. Additionally, when we have a lot of fetaures in our model, it is harder to explain the features with the strongest effect or what we call the **Big Picture**. Hence, we might want to choose fewer features in order to trade a worse in-sample variance for a better out-of-sample prediction.

[**Regularization**](https://en.wikipedia.org/wiki/Regularization_(mathematics)) is a method to shrink or drop coefficients/parameters from a model by imposing a penalty on their size. Since, it shrinks the actual least squares coeffiencts, it is also called **Shrinkage Method**. In this post, I will discuss two of the most common shrinkage methods - [**Ridge**](https://en.wikipedia.org/wiki/Tikhonov_regularization) and [**Lasso**](https://en.wikipedia.org/wiki/Lasso_(statistics)) regularization. 

## Setup

For starters, we will use the `Prostate Cancer` dataset from the [**Elements of Statistical Learning**](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576) book. If you want to more information about the dataset, it's available in *Chapter 1* of the book or [**here**](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.info.txt) and the dataset is available [**here**](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data)

If you want to follow along with some code, I have put a [`Jupyter` Notebook](https://github.com/kapilsh/ml-projects/blob/master/regularization/Regularization.ipynb) on **Github**.

```python
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from logorama import logger

from bokeh.io import output_notebook, curdoc
from bokeh.plotting import figure, show
from bokeh.themes import Theme
from bokeh.embed import components
from bokeh.models import Range1d
from bokeh.palettes import Spectral10

%matplotlib inline

plot_theme = Theme("./theme.yml")
output_notebook()

sns.set_style(style="dark")
```
Let's load the data and plot the pair-wise scatter plot for it.  

```python
prostate_data = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data", 
                            sep="\t", index_col=0)
train_idx = prostate_data["train"] == "T"
prostate_data.drop(columns="train", inplace=True)

fp = sns.pairplot(prostate_data, diag_kind="kde")
fp.fig.suptitle("Pair-wise Scatter Plot")
plt.show()
```

![png](/assets/images/prostate_multi_scatter.png)

We will need the [**L2-Norm**](http://mathworld.wolfram.com/L2-Norm.html) for the Linear Least Squares model, so let's implement that. 

```python
import scipy.optimize as so

def norm(betas, x_train, y_train):
    return np.linalg.norm(y_train - np.mean(y_train) - x_train.dot(betas))
```

<div class="box">
    <h4>About the dataset</h4>
	<p>
        The original data comes from a study by Stamey et. al. [1989], where they examined the relationship netween the level of prostate-specific antigen and number of clinical measures in men who were about to receive a radical prostatectomy. Features in model are: 
        <ul>
			<li>log cancer volume (lcavol)</li>
			<li>log prostate weight (lweight)</li>
            <li>age (age)</li>
            <li>log of the amount of benign prostatic hyperplasia (lbph)</li>
            <li>seminal vesicle invasion  (svi)</li>
			<li>log of capsular penetration (lcp)</li>
            <li>Gleason score (gleason)</li>
			<li>Percent of Gleason scores 4 or 5 (pgg45)</li>
		</ul>
        The response variable is:
        <ul>
            <li>log of prostate-specific antigen (lpsa)</li>
        </ul> 
    </p>
</div>

## Simple Linear Fit


First up, let's fit a least squares model to the data and get corresponding `standard error` and `z-score` estimates for each coefficient. We know that the linear regression fit is given by:

\begin{equation}
\hat y = \mathbf{X} \beta = \mathbf{X} (\mathbf{X^\intercal} \mathbf{X})^{-1} \mathbf{X^\intercal} y
\end{equation}
 
The variance-covariance matrix for least squares parameters is given by 

\begin{equation}
Var(\hat \beta) = (\mathbf{X^\intercal} \mathbf{X})^{-1} \sigma^2
\end{equation}

where $\sigma$ is the population standard deviation of $y_i$. We can estimate $\sigma^{2}$ by:

\begin{equation}
\hat \sigma^2 = \dfrac{1}{N - p -1} \sum\limits_{i=1}^N (y_i - \hat y_i)^2
\end{equation}

Finally, we can calculate the z-score as:

\begin{equation}
Z_{\beta_i} = \dfrac{\hat \beta_i - 0}{\hat \sigma_{\beta_i}}
\end{equation}

```python
y_data = prostate_data["lpsa"]

x_data = prostate_data.drop(columns="lpsa")
x_data = (x_data - x_data.mean()) / x_data.std() # Standardize
```


```python
x_train = x_data.loc[train_idx]
x_train = np.hstack([np.ones((len(x_train), 1)), x_train.values.copy()])
y_train = y_data.loc[train_idx]

betas = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)

y_train_hat = x_train.dot(betas) 
```


```python
dof = len(x_train) - len(betas)

mse = np.sum((y_train_hat - y_train) ** 2) / dof

betas_cov = np.linalg.inv(x_train.T.dot(x_train)) * mse
betas_se = np.sqrt(betas_cov.diagonal())
betas_z = (betas - 0) / betas_se

betas_estimate_table = pd.DataFrame({"Beta": betas, "SE": betas_se, "Z-Score": betas_z}, 
                                    index=np.append(["intercept"], x_data.columns))

logger.info(f"Degrees of Freedom: {dof}")
logger.info(f"MSE: {np.round(mse, 4)}")
logger.info(f"Beta Errors:\n{betas_estimate_table}")
```

```out
2018-09-08 16:09:32.860426 - [INFO] - {root:<module>:12} - Degrees of Freedom: 58
2018-09-08 16:09:32.865155 - [INFO] - {root:<module>:13} - MSE: 0.5074
2018-09-08 16:09:32.872027 - [INFO] - {root:<module>:14} - Beta Errors:
               Beta        SE    Z-Score
intercept  2.464933  0.089315  27.598203
lcavol     0.679528  0.126629   5.366290
lweight    0.263053  0.095628   2.750789
age       -0.141465  0.101342  -1.395909
lbph       0.210147  0.102219   2.055846
svi        0.305201  0.123600   2.469255
lcp       -0.288493  0.154529  -1.866913
gleason   -0.021305  0.145247  -0.146681
pgg45      0.266956  0.153614   1.737840
```
At a 95% confidence level, `z-score` greater/lesser than value of $\pm2$ is significant. 

> Out of the 9 features, 4 (`age`, `lcp`, `gleason`, `pgg45`) are not significant in our current model

Let's calculate the `MSE` for out-of-sample/test data

```python
x_test = x_data.loc[~train_idx]
x_test = np.hstack([np.ones((len(x_test), 1)), x_test.values.copy()])
y_test = y_data.loc[~train_idx]

y_test_hat = x_test.dot(betas) 


dof_test = len(x_test) - len(betas)
mse = np.sum((y_test_hat - y_test) ** 2) / dof_test

logger.info(f"MSE: {np.round(mse, 4)}")
```
```out
2018-09-08 16:09:33.139871 - [INFO] - {root:<module>:11} - MSE: 0.7447
```

We can see that the variance is much higher for the out-of-sample data, as expected in case of an overfit least squares model. 

Next, let's discuss how we can use regularization to help with this model.  

## Ridge Regression
Ridge Regression imposes an `L2-norm` based penalty on the sizes of coefficients given by $\beta^{\intercal} \beta$. We also introduce a complexity parameter $\lambda >= 0$ that controls the amount of shrinkage. Hence, the total shrinkage is given by $\lambda \beta^{\intercal} \beta$. 

> As the value of $\lambda$ gets larger, we get higher shrinkage

Our new regression coefficients are given by:

\begin{equation}
\hat \beta^{ridge} = \mathbf{argmin}_{\beta} \Bigg \\{ \sum _{i = 1}^N (y_i - \beta_0 - \sum _{j = 1}^p x _{ij} \beta_j )^2 + \lambda \sum _{j = 1}^p \beta _j^2 \Bigg \\}
\end{equation}

> Inputs need to be standardized to bring ridge coefficients to equivalent scale

Notice that $\beta_0$ is left out of the penalty term, since that will make the intercept depend on origin of $Y$. In other words, if we add a constant c to all $y_i$, our intercept should not change but if we include $\beta_0$ in penalty term, it will change. Hence, we reparameterize the model such that effective $y_i$ are given by $y_i - \bar y$ where $\bar y$ is given by $\dfrac{\sum _i^N y_i}{N}$. After reparameterization, $X$ has $p$ columns instead of $p+1$ and no constant term. Now, we can reqrite our `ridge` coefficients in matri form as:


\begin{equation}
\hat \beta^{ridge} = (\mathbf{X}^\intercal \mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\intercal y
\end{equation}

{% include bokeh/regularization/lambda_ridge_div.html %}
{% include bokeh/regularization/lambda_ridge_script.html %}

{% include bokeh/regularization/lambda_ridge_logl_div.html %}
{% include bokeh/regularization/lambda_ridge_logl_script.html %}

{% include bokeh/regularization/lambda_ridge_t_div.html %}
{% include bokeh/regularization/lambda_ridge_t_script.html %}

## Lasso Regression

{% include bokeh/regularization/lambda_lasso_div.html %}
{% include bokeh/regularization/lambda_lasso_script.html %}

{% include bokeh/regularization/lambda_lasso_l_div.html %}
{% include bokeh/regularization/lambda_lasso_l_script.html %}