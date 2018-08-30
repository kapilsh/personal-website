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
save_plot(p, "reg_by_h",  "/Users/kapilsharma/dev/git/kapilsh.github.io/_includes/bokeh/local_linear/")
```

{% include bokeh/local_linear/reg_by_h_div.html %}
{% include bokeh/local_linear/reg_by_h_script.html %}

{% include bokeh/local_linear/mse_vs_h_div.html %}
{% include bokeh/local_linear/mse_vs_h_script.html %}
## Cross Validation

### Leave One Out Cross Validation (LOOCV)

{% include bokeh/local_linear/loocv_mse_div.html %}
{% include bokeh/local_linear/loocv_mse_script.html %}


{% include bokeh/local_linear/loocv_fit_div.html %}
{% include bokeh/local_linear/loocv_fit_script.html %}

### K-Fold Cross Validation

{% include bokeh/local_linear/kcv_mse_div.html %}
{% include bokeh/local_linear/kcv_mse_script.html %}


{% include bokeh/local_linear/kcv_fit_div.html %}
{% include bokeh/local_linear/kcv_fit_script.html %}
