---
layout: post
title:  "Eigen3 Benchmarks"
description:  "Matrix Inversion using Eigen C++ Library"
image: /assets/images/posts/matrix.png
tags:
    - cpp
    - eigen3
    - spdlog
    - conan
comments: true
---

Eigen is super fast linear algebra library for C++. It provides almost all matrix / vector related operations and some extra `pandas` / `numpy` style functionality. Recently, one of my colleagues was looking for a linear algebra for C++ and I suggested using Eigen. During our conversation, we were discussing how fast are matrix inverse operation in Eigen, however the Eigen docs did not provide a satisfactory benchmarks for inversion. So, I decided to do a little test on my own.

# TL;DR Version

<div>
<table border="1" class="dataframe">
<thead>
  <tr>
    <th>Matrix Size</th>
    <th>Mean (in ns)</th>
    <th>Min (in ns)</th>
    <th>Max (in ns)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>1094.94</td>
    <td>0</td>
    <td>1.824e+06</td>
  </tr>
  <tr>
    <td>2</td>
    <td>1033.88</td>
    <td>0</td>
    <td>66000</td>
  </tr>
  <tr>
    <td>3</td>
    <td>1172.07</td>
    <td>1000</td>
    <td>99000</td>
  </tr>
  <tr>
    <td>4</td>
    <td>1262.35</td>
    <td>1000</td>
    <td>151000</td>
  </tr>
  <tr>
    <td>5</td>
    <td>1573.22</td>
    <td>1000</td>
    <td>134000</td>
  </tr>
  <tr>
    <td>6</td>
    <td>1747.54</td>
    <td>1000</td>
    <td>489000</td>
  </tr>
  <tr>
    <td>7</td>
    <td>1977.85</td>
    <td>1000</td>
    <td>51000</td>
  </tr>
  <tr>
    <td>8</td>
    <td>2076.58</td>
    <td>1000</td>
    <td>1.637e+06</td>
  </tr>
  <tr>
    <td>9</td>
    <td>2455.33</td>
    <td>2000</td>
    <td>110000</td>
  </tr>
  <tr>
    <td>10</td>
    <td>2759.75</td>
    <td>2000</td>
    <td>749000</td>
  </tr>
  <tr>
    <td>11</td>
    <td>3265.28</td>
    <td>3000</td>
    <td>209000</td>
  </tr>
  <tr>
    <td>12</td>
    <td>3354.84</td>
    <td>3000</td>
    <td>139000</td>
  </tr>
  <tr>
    <td>13</td>
    <td>4417.52</td>
    <td>3000</td>
    <td>3.187e+06</td>
  </tr>
  <tr>
    <td>14</td>
    <td>4542.06</td>
    <td>4000</td>
    <td>61000</td>
  </tr>
  <tr>
    <td>15</td>
    <td>5553.37</td>
    <td>5000</td>
    <td>462000</td>
  </tr>
  <tr>
    <td>16</td>
    <td>5469.77</td>
    <td>5000</td>
    <td>115000</td>
  </tr>
  <tr>
    <td>17</td>
    <td>7014.22</td>
    <td>6000</td>
    <td>788000</td>
  </tr>
  <tr>
    <td>18</td>
    <td>7344.36</td>
    <td>6000</td>
    <td>417000</td>
  </tr>
  <tr>
    <td>19</td>
    <td>8805.54</td>
    <td>8000</td>
    <td>184000</td>
  </tr>
  <tr>
    <td>20</td>
    <td>8824.95</td>
    <td>8000</td>
    <td>102000</td>
  </tr>
  <tr>
    <td>21</td>
    <td>10390.2</td>
    <td>9000</td>
    <td>2.865e+06</td>
  </tr>
  <tr>
    <td>22</td>
    <td>11314.2</td>
    <td>10000</td>
    <td>343000</td>
  </tr>
  <tr>
    <td>23</td>
    <td>13065.2</td>
    <td>12000</td>
    <td>728000</td>
  </tr>
  <tr>
    <td>24</td>
    <td>12958.2</td>
    <td>12000</td>
    <td>189000</td>
  </tr>
  <tr>
    <td>25</td>
    <td>15338.9</td>
    <td>14000</td>
    <td>1.149e+06</td>
  </tr>
</tbody>

</table>
</div>

# For the Curious Ones

I have uploaded the cmake project for this test on my [Github](https://github.com/kapilsh/mini-projects/tree/master/cpp/eigen-benchmarks). If you want to explore and play around with the project, please keep reading.

In this project, I have also used [conan](https://conan.io/), which is a python-based open-source C++ package manager. Recently, `conan` community released the 1.0 version of the package. We can install it by simply using pip.  

```bash
pip install conan
```

After instaling `conan`, you will need to add a couple of conan repositories.

```bash
conan remote add conan-community "https://api.bintray.com/conan/conan-community/conan" # For spdlog
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan # For Eigen3
```

After adding the repositories, we can setup a conanfile.txt in the root directory of the project.

```
[requires]
eigen/3.3.4@conan/stable
spdlog/0.14.0@bincrafters/stable

[generators]
cmake
```
`conan` will generate a .cmake file, which will have paths to all the libraries that we use in our project.

On my Mac, I use the latest gcc 7.3 installed from `Homebrew`.

```
brew install gcc
```

If `Homebrew` gcc is already installed, we can upgrade it using:

```
brew upgrade gcc
```

Below is how my CMakeLists.txt file looks like.

```cmake

cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 17)

file(COPY ${CMAKE_SOURCE_DIR}/conanfile.txt DESTINATION .)

execute_process(COMMAND conan install . --settings os=Macos -g cmake --profile gcc RESULT_VARIABLE CONAN_EXIT_CODE)

include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

if (APPLE)
    set(GCC_ROOT_DIR /usr/local/bin)

    set(CMAKE_C_COMPILER   ${GCC_ROOT_DIR}/gcc-7)
    set(CMAKE_CXX_COMPILER ${GCC_ROOT_DIR}/g++-7)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

project(eigen-benchmarks)

conan_define_targets()

include_directories(${CONAN_INCLUDE_DIRS_EIGEN})
include_directories(${CONAN_INCLUDE_DIRS_SPDLOG})

set(SOURCE_FILES main.cpp)
add_executable(eigen-benchmarks ${SOURCE_FILES})
install(TARGETS eigen-benchmarks DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
```
The main program is below:

```cpp
#include <iostream>
#include "numeric"
#include "chrono"
#include <time.h>
#include <Eigen/Dense>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/bundled/ostream.h"

namespace spd = spdlog;
constexpr int MAX_MATRIX_SIZE = 25;
constexpr int TOTAL_ITERATIONS = 100000;
constexpr int IGNORE_COUNT = 10;

double_t calc_time(int32_t size);

int main()
{
    srand(static_cast<unsigned int>(time(nullptr)));
    auto console = spd::stdout_color_mt("console");
    console->info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
    console->info("Eigen Benchmarks (in ns)");
    console->info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

    std::map<int32_t, double_t> averages;
    std::map<int32_t, double_t> maxs;
    std::map<int32_t, double_t> mins;

    std::cout << "Matrix Size | Mean | Min | Max" << std::endl;
    std::cout << "------------|------|-----|----" << std::endl;

    for (int i = 0; i < MAX_MATRIX_SIZE; ++i)
    {
        std::vector<double_t> timings;
        for (int j = 0; j < TOTAL_ITERATIONS; ++j)
        {
            if (j >= IGNORE_COUNT) {
                auto time_taken = calc_time(i + 1);
                timings.push_back(time_taken);
            }
        }

        auto avg = (std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size());
        auto minimum = *std::min_element(std::begin(timings), std::end(timings));
        auto maximum = *std::max_element(std::begin(timings), std::end(timings));

        std::cout << i + 1 << " | " << avg << " | " << minimum << " | " << maximum << std::endl;
    }

    console->info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
}

double_t calc_time(int32_t size)
{
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(size, size);
    auto invertible = m * m.transpose();
    auto begin = std::chrono::steady_clock::now();
    Eigen::MatrixXd m_inv = m.inverse();
    auto end = std::chrono::steady_clock::now();
    auto time_diff = end - begin;
    auto time_taken = static_cast<double_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_diff).count());
    return time_taken;
}

```
