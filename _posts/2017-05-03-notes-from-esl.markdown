---
layout: post
title:  "Notes from ESL - Chapter 2"
date:   2017-05-01 22:00:00
tags:
    - machinelearning
    - theory
image: /images/ai.png
comments: true
---

The book would be primarily focus on 4 data-sets:

- Email Spam
- Prostate Cancer
- Handwritten Digits
- DNA Expression Microarrays

## Introduction

**Supervised Learning**: Use inputs to predict the values of outputs

- **Inputs** are also called features, predictors, and independent variables
- **Outputs** are also called responses, and dependent variables

Three types of variables:

- Quantitative
- Categorical
- Ordinal

Qualitative variables most times are represented by numbers such as ${ 0, 1 }$ or ${ -1, 1 }$.

The most used technique is using **dummy variables**, where k-level qualitative variable is represented by a vector of k binary variables.

## Linear Model versus k-Nearest Neighbors

### Linear Model:

- Stable
- Lots of assumptions
- Inaccurate

$\beta_o$ is considered the bias in the model.

### k-NN
- Accurate
- Unstable
