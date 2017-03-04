---
layout: post
title:  "Geometry of Regression"
date:   2017-03-01 22:00:00
tags:
    - python
    - machinelearning
image: /ipynb/geometric-representation-of-regression_files/regression.png
---

A picture is worth a thousand words. The below graphic captures the essense of Multiple Regression very aptly.

![Regression Picture](/ipynb/geometric-representation-of-regression_files/regression.png)

[This post](http://stats.stackexchange.com/questions/123651/geometric-interpretation-of-multiple-correlation-coefficient-r-and-coefficient) on Stack Exchange gives a great description of the graphic and geometric representation of regression problems.

Source: [Stack Exchange](http://stats.stackexchange.com/questions/123651/geometric-interpretation-of-multiple-correlation-coefficient-r-and-coefficient)

**Let's see this in action.**


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.style.use("ggplot")
```


```python
x = np.array([1.0, 2, 3])
y = np.array([2, 2.5, 5])
```


```python
fig = plt.figure()
plt.scatter(x, y)
plt.xlim((0, 5))
plt.ylim((0, 6))
plt.title("Y vs X")
plt.xlabel("X")
plt.ylabel("Y")
plt.gcf().set_size_inches(10, 5)
plt.show()
```

![png](/ipynb/geometric-representation-of-regression_files/geometric-representation-of-regression_9_0.png)


We know that the linear regression coefficients are given by:

\begin{equation}
\beta = (\mathbf{X^\intercal} \mathbf{X})^{-1} \mathbf{X^\intercal} y
\end{equation}


```python
X = np.append(np.ones(len(x)), x).reshape(2, 3).T
```


```python
X
```

array([[ 1.,  1.],
       [ 1.,  2.],
       [ 1.,  3.]])

```python
beta = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y).T)
beta
```

array([ 0.16666667,  1.5       ])

```python
b_0 = beta[0]
b_1 = beta[1]
```

Let us calculate our $\hat y$


```python
y_hat = np.matmul(X, beta)
```


```python
fig = plt.figure()
plt.scatter(x, y)
plt.xlim((0, 5))
plt.ylim((0, 6))
plt.title("Y vs X | Regression Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(np.append([0], x), np.append([b_0], y_hat), color='blue')
plt.gcf().set_size_inches(10, 5)
plt.show()
```

![png](/ipynb/geometric-representation-of-regression_files/geometric-representation-of-regression_17_0.png)


```python
res = y - y_hat
res
```
array([ 0.33333333, -0.66666667,  0.33333333])

Now the average vector $\bar y$


```python
y_bar = np.mean(y) * np.ones(len(y))
y_bar
```

array([ 3.16666667,  3.16666667,  3.16666667])

```python
base = y_hat - y_bar
base
```

array([ -1.50000000e+00,   1.33226763e-15,   1.50000000e+00])

```python
dot_product = np.dot(base, res)
dot_product
```

0.0

Hence, we see that dot product between residual $\epsilon$ and $\hat y - \bar y$ equals $0$

```python
hyp = y - y_bar
hyp
```

array([-1.16666667, -0.66666667,  1.83333333])

Let's check the `Pythagoras Theorem`


```python
np.power(np.linalg.norm(hyp), 2) - np.power(np.linalg.norm(base), 2) - np.power(np.linalg.norm(res), 2)
```

3.3306690738754696e-16

which is practically $0$

## Summary

Through this post, I wanted to demonstrate how we can interpret linear/multiple regression geometrically. Also, I also demonstrated how to solve a linear model through linear algebra.
