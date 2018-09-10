---
layout: post
title: C++ 17 Fetaures
description: List of essential C++17 features 
date: 2018-09-05 08:00:00
image: /assets/images/ISO_Cpp.svg
tags:
    - cpp
    - c++17
    - language
comments: true
---

Here is a list of essential C++ features

## Structural Bindings

```cpp 
// Binding Array

int a[2] = {1,2};
 
auto [x,y] = a; // creates e[2], copies a into e, then x refers to e[0], y refers to e[1]
auto& [xr, yr] = a; // xr refers to a[0], yr refers to a[1]
```

## Selection Statements with Initializer

## Fold expressions
## Compile-time conditional statments
## Class template argument deduction (CTAD)
## auto non-type template parameters
## inline variables
## constexpr lambdas
## Guaranteed copy elision
## Nested namespace definitions

## string_view
## optional
## variant
## any
## Parallel algorithms
## New algorithms
## Filesystem support
## Polymorphic Allocators
## Improved insertion and splicing for associative containers
## Variable templates for metafunctions
## Boolean logic metafunctions