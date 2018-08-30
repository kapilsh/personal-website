---
layout: post
title: Local Linear Regression
description: Moving from Locally Weighted Constants to Functions 
date: 2018-08-26 08:00:00
image: /assets/images/Localregressionsmoother.svg
tags:
    - python
    - machine learning
    - kernel methods
comments: true
bokeh: true
---

I previously wrote a [post](http://www.sharmakapil.com/2018/08/26/kernel-smoothing.html) about **Kernel Smoothing** and how it can be used to fit any relationship non-parametrically. In this port I will extend on that idea and try to mitigate the disadvantages of kernel smoothing using **Local Linear Regression**. 

## Setup

I generated some data in my previous [post](http://www.sharmakapil.com/2018/08/26/kernel-smoothing.html) and I will reuse the same data. The data was generated from the function $\mathbf{y = f(x) = sin(4x) + 2}$ with some gaussian noise and here's how it looks like:

{% include bokeh/local_linear/yvx_div.html %}
{% include bokeh/local_linear/yvx_script.html %}

## Local Linear Regression

As I mentioned in the previous article, kernel smoothing has issues in out-of-sample prediction on the edges and in sparse regions. In **Local Linear Regression**, we try to reduce this bias to first order by fitting straight lines instead of local constants. 

Locally weighted regression solves a weighted least squares problem at each out-of-sample point $x_0$, geiven by:

\begin{equation}
\mathbf{Equation Goes Here}
\end{equation}

```python
def predict(x_test, x_train, y_train, h):
    if len(x_train) != len(y_train):
        raise ValueError("X and Y Should have same length")
    B = np.array([np.ones(len(x_train)), x_train]).T
    y_hat = []
    for x0 in x_test:
        W = np.diag(gaussian_kernel(x_train , x0, h))
        y_hat.append(np.array([1, x0]).T.dot(
            np.linalg.inv(B.T.dot(W).dot(B))).dot(
            B.T).dot(W).dot(y_train))
    
    return np.array(y_hat)
```

```python
h_values = [0.01, 0.1, 1]
colors = ["#A6E22E", "#FD971F", "#AE81FF"]
```

```python
p = figure(plot_width=800, plot_height=400)
p.circle(x, y, size=10, alpha=0.2, color="#66D9EF", legend="y")
p.line(x, f(x, 2), color="#F92672", line_width=3, legend="Actual", line_dash="dashed")

for idx, h in enumerate(h_values):
    p.line(x, predict(x, x, y, h), color=colors[idx], line_width=2, legend="y_hat (h={})".format(h))
    
p.title.text = "Local Linear Regression (Gaussian Kernel)"
p.xaxis.axis_label = "X"
p.yaxis.axis_label = "Y"

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```

{% include bokeh/local_linear/reg_by_h_div.html %}
{% include bokeh/local_linear/reg_by_h_script.html %}


```python
h_trial = 0.1
x_trials = np.linspace(domain[0], domain[1], 6)

def local_coeffs(x_0, x_train, y_train, h):
    if len(x_train) != len(y_train):
        raise ValueError("X and Y Should have same length")
    
    B = np.array([np.ones(len(x_train)), x_train]).T
    W = np.diag(gaussian_kernel(x_train , x_0, h_trial))
    return np.linalg.inv(B.T.dot(W).dot(B)).dot(B.T).dot(W).dot(y_train)

coeffs = [local_coeffs(x_0, x, y, h_trial) for x_0 in x_trials]
print(coeffs)
```

```
[array([ 1.10375711, -0.24977984]), 
array([ 1.93831019,  3.12814681]), 
array([ 2.04711505,  3.62427698]), 
array([ 2.58841427,  1.23608707]), 
array([ 3.96096982, -2.14935988]), 
array([ 5.16166187, -4.00951684])]
```


```python
p = figure(plot_width=800, plot_height=400)
p.circle(x, y, size=10, alpha=0.2, color="#66D9EF")

x_left = []
x_right = []
y_left = []
y_right = []

y_fits = []

for cs, x_0 in zip(coeffs, x_trials):
    x_left.append(x_0 - 0.1)
    x_right.append(x_0 + 0.1)
    y_left.append(np.array([1, x_0 - 0.1]).dot(cs))
    y_right.append(np.array([1, x_0 + 0.1]).dot(cs))
    y_fits.append(np.array([1, x_0]).dot(cs))
    

p.line(x, predict(x, x, y, h_trial), color="#FD971F", line_width=2, line_dash="dashed", legend="y_hat")
p.segment(x0=x_left, y0=y_left, x1=x_right, y1=y_right, color="#A6E22E", line_width=1, legend="Local Fit")
p.circle(x_trials, y_fits, size=5, color="#A6E22E", legend="Local Fit")
    
p.title.text = "Local Linear Fits"
p.xaxis.axis_label = "X"
p.yaxis.axis_label = "Y"

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```
{% include bokeh/local_linear/local_lines_div.html %}
{% include bokeh/local_linear/local_lines_script.html %}

```python
h_range = np.linspace(0.01, 0.2, 20)
mses = [np.mean(np.power(y - predict(x, x, y, h), 2)) for h in h_range]
```


```python
p = figure(plot_width=800, plot_height=400)
p.circle(x=h_range, y=mses, size=10, color="#66D9EF")
p.line(x=h_range, y=mses, color="#66D9EF", line_width=3)

p.title.text = "MSE vs Bandwidth"
p.xaxis.axis_label = "Bandwidth"
p.yaxis.axis_label = "MSE"

p.y_range = Range1d(0.8, 1.2)  # from bokeh.models import Range1d

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```
{% include bokeh/local_linear/mse_vs_h_div.html %}
{% include bokeh/local_linear/mse_vs_h_script.html %}

## Cross Validation

### Leave One Out Cross Validation (LOOCV)

```python
mse_values = []

for h in h_range:
    errors = []
    for idx, val in enumerate(x):
        x_test = np.array([val])
        y_test = np.array([y[idx]])
        x_train = np.append(x[:idx], x[idx+1:])
        y_train = np.append(y[:idx], y[idx+1:])
        assert len(x_train) == data_size - 1
        y_test_hat = predict(x_test, x_train, y_train, h)
        errors.append((y_test_hat - y_test)[0])
    mse_values.append(np.mean(np.power(errors, 2)))
```


```python
p = figure(plot_width=800, plot_height=400)
p.circle(x=h_range, y=mse_values, size=10, color="#66D9EF")
p.line(x=h_range, y=mse_values, color="#66D9EF", line_width=3)

p.title.text = "Cross Validation - LOOCV - MSE vs Bandwidth"
p.xaxis.axis_label = "Bandwidth"
p.yaxis.axis_label = "MSE"

p.y_range = Range1d(0.8, 1.2)

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```

{% include bokeh/local_linear/loocv_mse_div.html %}
{% include bokeh/local_linear/loocv_mse_script.html %}

```python
h_optimal = h_range[np.argmin(mse_values)]
print(h_optimal)
```
`Out[1] : 0.09`

```python
p = figure(plot_width=800, plot_height=400)
p.circle(x, y, size=10, alpha=0.2, color="#66D9EF", legend="y")
p.line(x, f(x, 2), color="#F92672", line_width=3, legend="Actual", line_dash="dashed")

p.line(x, predict(x, x, y, h_optimal), color="#A6E22E", line_width=2, legend="y_hat (h={})".format(h_optimal))
    
p.title.text = "Cross Validation - LOOCV - Optimal Fit"
p.xaxis.axis_label = "x"
p.yaxis.axis_label = "f(x)"

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```

{% include bokeh/local_linear/loocv_fit_div.html %}
{% include bokeh/local_linear/loocv_fit_script.html %}

### K-Fold Cross Validation


```python
num_folds = 10
num_tries = 5

fold_indices  = np.arange(num_folds)
mse_values = []

for h in h_range:
    trial_mses = []
    for trial in np.arange(num_tries):
        x_splits, y_splits = split_k_fold(x, y, num_folds)
        mses = []
        for idx in fold_indices:
            test_idx = idx
            train_idx = np.setdiff1d(fold_indices, [idx])
            train_x, test_x, train_y, test_y = (np.concatenate(x_splits[train_idx]), 
                                                x_splits[test_idx], 
                                                np.concatenate(y_splits[train_idx]), 
                                                y_splits[test_idx])
            test_y_hat = predict(test_x, train_x, train_y, h)
            mses.append(np.mean(np.power(test_y_hat - test_y, 2)))
        trial_mses.append(np.mean(mses))
    mse_values.append(np.mean(trial_mses))
```


```python
p = figure(plot_width=800, plot_height=400)
p.circle(x=h_range, y=mse_values, size=10, color="#66D9EF")
p.line(x=h_range, y=mse_values, color="#66D9EF", line_width=3)

p.title.text = "Cross Validation - K-Fold - MSE vs Bandwidth"
p.xaxis.axis_label = "Bandwidth"
p.yaxis.axis_label = "MSE"

p.y_range = Range1d(0.8, 1.2)

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```

{% include bokeh/local_linear/kcv_mse_div.html %}
{% include bokeh/local_linear/kcv_mse_script.html %}


```python
h_optimal = h_range[np.argmin(mse_values)]
print(h_optimal)
```

`Out[2] : 0.06`

```python
p = figure(plot_width=800, plot_height=400)
p.circle(x, y, size=10, alpha=0.2, color="#66D9EF", legend="y")
p.line(x, f(x, 2), color="#F92672", line_width=3, legend="Actual", line_dash="dashed")

p.line(x, predict(x, x, y, h_optimal), color="#A6E22E", line_width=2, legend="y_hat (h={})".format(h_optimal))
    
p.title.text = "Cross Validation - K-Fold - Optimal Fit"
p.xaxis.axis_label = "x"
p.yaxis.axis_label = "f(x)"

curdoc().clear()
doc = curdoc()
doc.theme = plot_theme
doc.add_root(p)
show(p)
```

{% include bokeh/local_linear/kcv_fit_div.html %}
{% include bokeh/local_linear/kcv_fit_script.html %}
