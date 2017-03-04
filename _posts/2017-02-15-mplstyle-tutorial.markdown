---
layout: post
title:  "Tutorial introduction to mplstyle stylesheets"
date:   2017-02-15 08:00:00
tags:
    - python
    - matplotlib
image: /images/mt1.png
---

One of my favorite libraries in Python is [matplotlib](http://matplotlib.org/). Despite the many options we Python programmers have, I still find the simplicity and rawness of `matplotlib` great to work with. I recently discovered the usefulness of matplotlib stylesheets and how they can help in customizing your plots. In this tutorial, I have laid out the process that I used to get up and running with stylesheets.

## Introduction: Matplotlib Stylesheets

A good amount of my workflow involves creating reproducible research in [jupyter-notebooks](http://jupyter.readthedocs.io/en/latest/content-quickstart.html). Notebooks are great for adhoc research as well as for creating publication quality pdf reports or web based dashboards. In fact, I have produced this particular webpage in a [notebook](/ipynb/mplstyle-tutorial.ipynb).

While matplotlib comes with a lot of different stylesheets, I have run into scenarios where I wanted to create a style that resembled a particular color-scheme. This led me to explore [matplotlib stylesheets](http://matplotlib.org/users/style_sheets.html#defining-your-own-style).

Throughout this post, I will demonstrate how to use `matplotlib` stylesheets to customize plots for your own needs.  

## Data Creation

First, let's create some random data using [Pandas](http://pandas.pydata.org/) and [NumPy](http://www.numpy.org/)


```python
import pandas as pd
import numpy as np
```


```python
rand_data = np.random.randn(1000 * 3).reshape(1000, 3)
returns = pd.DataFrame(rand_data, columns=list("ABC"),
                       index=pd.date_range(start="2016-12-01 08:30:00",
                                           freq="1s", periods=1000))

prices = returns.cumsum() + 100
```


```python
prices[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-12-01 08:30:00</th>
      <td>100.232142</td>
      <td>99.682473</td>
      <td>99.521759</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:01</th>
      <td>99.978837</td>
      <td>99.446970</td>
      <td>100.423348</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:02</th>
      <td>100.756137</td>
      <td>100.772167</td>
      <td>99.999918</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:03</th>
      <td>101.025152</td>
      <td>99.063761</td>
      <td>100.486253</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:04</th>
      <td>99.177024</td>
      <td>99.689774</td>
      <td>101.648877</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:05</th>
      <td>100.318996</td>
      <td>99.723182</td>
      <td>102.756434</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:06</th>
      <td>98.370779</td>
      <td>101.317465</td>
      <td>101.901374</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:07</th>
      <td>98.293057</td>
      <td>102.366967</td>
      <td>101.768990</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:08</th>
      <td>98.319536</td>
      <td>102.300582</td>
      <td>101.785594</td>
    </tr>
    <tr>
      <th>2016-12-01 08:30:09</th>
      <td>98.133255</td>
      <td>102.308406</td>
      <td>103.178428</td>
    </tr>
  </tbody>
</table>
</div>



## Basic matplotlib plotting

Let's plot this data as a time-series keeping out-of-the-box settings for [matplotlib](http://matplotlib.org/).


```python
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
%matplotlib inline
```


```python
def plot_ts(df):
    fig, ax = plt.subplots()
    ax.plot_date(df.index.values, prices, '-')
    ax.set_ylabel("Price")
    ax.set_xlabel("Time")
    plt.suptitle("A Time Series Chart")
    plt.show()  
```


```python
plot_ts(prices)
```


![png](/ipynb/mplstyle-tutorial_files/mplstyle-tutorial_12_0.png)


## Available stylesheets

`matplotlib` comes with a few stylesheet options.

[seaborn](http://seaborn.pydata.org/) is another well-known plotting package for Python and is built as a wrapper on top of `matplotlib`. It installs its themes with the `matplotlib` themes.

Let's look at the available options for stylesheets.


```python
plt.style.available
```




    [u'seaborn-darkgrid',
     u'seaborn-notebook',
     u'classic',
     u'seaborn-ticks',
     u'grayscale',
     u'bmh',
     u'seaborn-talk',
     u'dark_background',
     u'ggplot',
     u'fivethirtyeight',
     u'seaborn-colorblind',
     u'seaborn-deep',
     u'seaborn-whitegrid',
     u'seaborn-bright',
     u'seaborn-poster',
     u'seaborn-muted',
     u'seaborn-paper',
     u'seaborn-white',
     u'seaborn-pastel',
     u'seaborn-dark',
     u'seaborn',
     u'seaborn-dark-palette']



## Using matplotlib stylesheets

We will use the `matplotlib` themes and see how they change the graph.

### ggplot


```python
plt.style.use("ggplot")
plot_ts(prices)
```


![png](/ipynb/mplstyle-tutorial_files/mplstyle-tutorial_19_0.png)


### bmh


```python
plt.style.use("bmh")
plot_ts(prices)
```


![png](/ipynb/mplstyle-tutorial_files/mplstyle-tutorial_21_0.png)


## Customization

If one of these options does not work for you, you are in the same boat as I was in one of my research projects.

First thing I did was find a good palette of colors. One great resource to find or create a good pallete is [Paletton](http://paletton.com/).

Here is a good article on [how and why to choose a good color palette](http://www.makeuseof.com/tag/pick-color-scheme-like-pro/).

![Palleton 1](/ipynb/mplstyle-tutorial_files/paletton1.png)

![Palleton 12](/ipynb/mplstyle-tutorial_files/paletton2.png)

Using the color-scheme, I have created a new mplstyle document, which you can download from [kapilsh.mplstyle](/ipynb/kapilsh.mplstyle).

The document looks like this:

```yaml

patch.linewidth: 0.5
patch.facecolor: 348ABD
patch.edgecolor: EEEEEE
patch.antialiased: True

font.size: 10.0

axes.facecolor: D5D5D5
axes.edgecolor: black
axes.linewidth: 1
axes.grid: True
axes.titlesize: large
axes.labelsize: medium
axes.labelcolor: black
axes.axisbelow: True

axes.prop_cycle: cycler('color', ['5E4F6D', 'A18271', '4B6B61', 'A19E71', '623F2C', '312142','1D4136', '625F2C'])

xtick.color: black
xtick.direction: out

ytick.color: black
ytick.direction: out

grid.color: white
grid.linestyle: -

figure.facecolor: white
figure.edgecolor: 0.75
figure.figsize: 12, 10
figure.autolayout: False

```

For the property keys, you can reference your [matplotlibrc](http://matplotlib.org/users/customizing.html) file. It is typically imstalled in `~/.config/matplotlib/`.

## Sample plots

Let's use this stylesheet to create some sample graphs.


```python
plt.style.use("kapilsh.mplstyle")
plot_ts(prices)
```


![png](/ipynb/mplstyle-tutorial_files/mplstyle-tutorial_30_0.png)



```python
fig, ax_arr = plt.subplots(ncols=3, nrows=1, sharey=True)
ax = ax_arr[0]
n, bins, patches = ax.hist(returns["A"], 50, normed=1, histtype='stepfilled')
ax.set_ylabel("Frequency")
ax.set_xlabel("Returns")
ax.set_title("A")
ax = ax_arr[1]
n, bins, patches = ax.hist(returns["B"], 50, normed=1, histtype='stepfilled')
ax.set_xlabel("Returns")
ax.set_title("B")
ax = ax_arr[2]
n, bins, patches = ax.hist(returns["C"], 50, normed=1, histtype='stepfilled')
ax.set_xlabel("Returns")
ax.set_title("C")
plt.suptitle("Distribution of Returns")
plt.gcf().set_size_inches(15, 5)
plt.show()
```


![png](/ipynb/mplstyle-tutorial_files/mplstyle-tutorial_31_0.png)



```python
returns_sample = returns[:200]

fig, ax = plt.subplots()
ax.plot_date(returns_sample.index.values, returns_sample["A"], '-')
ax.plot_date(returns_sample.index.values, returns_sample["B"], '-')
ax.plot_date(returns_sample.index.values, returns_sample["C"], '-')
ax.set_ylabel("Returns")
ax.set_xlabel("Time")
ax.set_title("Returns Time Series")
plt.gcf().set_size_inches(15, 5)
plt.show()
```


![png](/ipynb/mplstyle-tutorial_files/mplstyle-tutorial_32_0.png)


## Summary

In this post, I have demonstrated how to use `matplotlib` stylesheets to customize `matplotlib` plots.

I used **Paletton** to create a color-scheme.

Finally, I generated a new mplstyle stylesheet document and showed how to load it into your matplotlib environment to customize plots.
