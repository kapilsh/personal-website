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

{% include bokeh/local_linear/reg_by_h_div.html %}
{% include bokeh/local_linear/reg_by_h_script.html %}

{% include bokeh/local_linear/mse_vs_h_div.html %}
{% include bokeh/local_linear/mse_vs_h_script.html %}

## Cross Validation

### Leave One Out Cross Validation (LOOCV)

{% include bokeh/local_linear/loocv_mse_div.html %}
{% include bokeh/local_linear/loocv_mse_script.html %}

### K-Fold Cross Validation

