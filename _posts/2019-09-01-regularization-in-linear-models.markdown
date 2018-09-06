---
layout: post
title: Regularization in Linear Models
description: Ridge, Lasso, and Elastic Net 
date: 2018-09-01 08:00:00
image: /assets/images/ridge_lasso.png
tags:
    - python
    - machine-learning
    - regularization
comments: true
bokeh: true
---



```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
%matplotlib inline
```

```python
prostate_data = pd.read_csv("prostate", sep="\t", index_col=0)
idx = prostate_data.train == "T"
train_data = prostate_data.loc[idx]
test_data = prostate_data.loc[~idx]
train_data.drop(columns=["train"], inplace=True)
test_data.drop(columns="train", inplace=True)
```

```python
sns.pairplot(train_data, diag_kind="kde")
```

![png](/assets/images/prostate_multi_scatter.png)

## Ridge Regression

{% include bokeh/regularization/lambda_ridge_div.html %}
{% include bokeh/regularization/lambda_ridge_script.html %}

## Lasso Regression