import React from "react";
import { Typography, Alert, Table } from "antd";
import { GithubFilled } from "@ant-design/icons";
import { BashSnippet } from "../snippets/BashSnippet";
import { CppSnippet } from "../snippets/CppSnippet";

const { Title, Paragraph } = Typography;

const tableData = [
  {
    MatrixSize: 1,
    Mean: 1094.94,
    Min: 0,
    Max: 1824000,
  },
  {
    MatrixSize: 2,
    Mean: 1033.88,
    Min: 0,
    Max: 66000,
  },
  {
    MatrixSize: 3,
    Mean: 1172.07,
    Min: 1000,
    Max: 99000,
  },
  {
    MatrixSize: 4,
    Mean: 1262.35,
    Min: 1000,
    Max: 151000,
  },
  {
    MatrixSize: 5,
    Mean: 1573.22,
    Min: 1000,
    Max: 134000,
  },
  {
    MatrixSize: 6,
    Mean: 1747.54,
    Min: 1000,
    Max: 489000,
  },
  {
    MatrixSize: 7,
    Mean: 1977.85,
    Min: 1000,
    Max: 51000,
  },
  {
    MatrixSize: 8,
    Mean: 2076.58,
    Min: 1000,
    Max: 1637000,
  },
  {
    MatrixSize: 9,
    Mean: 2455.33,
    Min: 2000,
    Max: 110000,
  },
  {
    MatrixSize: 10,
    Mean: 2759.75,
    Min: 2000,
    Max: 749000,
  },
  {
    MatrixSize: 11,
    Mean: 3265.28,
    Min: 3000,
    Max: 209000,
  },
  {
    MatrixSize: 12,
    Mean: 3354.84,
    Min: 3000,
    Max: 139000,
  },
  {
    MatrixSize: 13,
    Mean: 4417.52,
    Min: 3000,
    Max: 3187000,
  },
  {
    MatrixSize: 14,
    Mean: 4542.06,
    Min: 4000,
    Max: 61000,
  },
  {
    MatrixSize: 15,
    Mean: 5553.37,
    Min: 5000,
    Max: 462000,
  },
  {
    MatrixSize: 16,
    Mean: 5469.77,
    Min: 5000,
    Max: 115000,
  },
  {
    MatrixSize: 17,
    Mean: 7014.22,
    Min: 6000,
    Max: 788000,
  },
  {
    MatrixSize: 18,
    Mean: 7344.36,
    Min: 6000,
    Max: 417000,
  },
  {
    MatrixSize: 19,
    Mean: 8805.54,
    Min: 8000,
    Max: 184000,
  },
  {
    MatrixSize: 20,
    Mean: 8824.95,
    Min: 8000,
    Max: 102000,
  },
  {
    MatrixSize: 21,
    Mean: 10390.2,
    Min: 9000,
    Max: 2865000,
  },
  {
    MatrixSize: 22,
    Mean: 11314.2,
    Min: 10000,
    Max: 343000,
  },
  {
    MatrixSize: 23,
    Mean: 13065.2,
    Min: 12000,
    Max: 728000,
  },
  {
    MatrixSize: 24,
    Mean: 12958.2,
    Min: 12000,
    Max: 189000,
  },
  {
    MatrixSize: 25,
    Mean: 15338.9,
    Min: 14000,
    Max: 1149000,
  },
];

const columns = [
  {
    title: "Matrix Size",
    dataIndex: "MatrixSize",
    key: "MatrixSize",
  },
  {
    title: "Mean (in ns)",
    dataIndex: "Mean",
    key: "Mean",
  },
  {
    title: "Min (in ns)",
    dataIndex: "Min",
    key: "Min",
  },
  {
    title: "Max (in ns)",
    dataIndex: "Max",
    key: "Max",
  },
];

class EigenBenchmarks extends React.Component {
  render() {
    return (
      <>
        <Typography>
          <Paragraph>
            Eigen is super fast linear algebra library for C++. It provides
            almost all matrix / vector related operations and some extra pandas
            / numpy style functionality. Recently, one of my colleagues was
            looking for a linear algebra for C++ and I suggested using Eigen.
            During our conversation, we were discussing how fast are matrix
            inverse operation in Eigen, however the Eigen docs did not provide a
            satisfactory benchmarks for inversion. So, I decided to do a little
            test on my own.
          </Paragraph>
          <Title level={3}>TL;DR Version</Title>
          <Table dataSource={tableData} columns={columns} />
          <Title level={3}>For the Curious Ones</Title>
          <Paragraph>
            I have uploaded the cmake project for this test on my{" "}
            <a
              href={
                "https://github.com/kapilsh/mini-projects/tree/master/cpp/eigen-benchmarks"
              }
            >
              <GithubFilled /> Github
            </a>
            . If you want to explore and play around with the project, please
            keep reading. In this project, I have also used{" "}
            <a href={"https://conan.io/"}>conan</a>, which is a python-based
            open-source C++ package manager. Recently, conan community released
            the 1.0 version of the package. We can install it by simply using
            pip.
          </Paragraph>
        </Typography>
        <BashSnippet
          text={`pip install conan

# After instaling conan, you will need to add a couple of conan repositories.
conan remote add conan-community "https://api.bintray.com/conan/conan-community/conan" # For spdlog
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan # For Eigen3`}
          hideLineNumbers
        />
        <br />
        <Typography>
          <Paragraph>
            After adding the repositories, we can setup a conanfile.txt in the
            root directory of the project.
          </Paragraph>
        </Typography>
        <BashSnippet
          text={`[requires]
eigen/3.3.4@conan/stable
spdlog/0.14.0@bincrafters/stable

[generators]
cmake`}
          hideLineNumbers
        />
        <br />
        <Alert
          message="NOTE"
          description="conan will generate a .cmake file, which will have paths to
          all the libraries that we use in our project. On my Mac, I use the
          latest gcc 7.3 installed from Homebrew"
          type="info"
          showIcon
        />
        <br />
        <BashSnippet
          text={"brew upgrade gcc && brew install gcc"}
          hideLineNumbers
        />
        <br />
        <Typography>
          <Paragraph>Below is how my CMakeLists.txt file looks like.</Paragraph>
        </Typography>
        <BashSnippet
          text={`cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 17)
          
file(COPY \${CMAKE_SOURCE_DIR}/conanfile.txt DESTINATION .)
          
execute_process(COMMAND conan install . --settings os=Macos -g cmake --profile gcc RESULT_VARIABLE CONAN_EXIT_CODE)
          
include(\${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
          
if (APPLE)
    set(GCC_ROOT_DIR /usr/local/bin)
    
    set(CMAKE_C_COMPILER   \${GCC_ROOT_DIR}/gcc-7)
    set(CMAKE_CXX_COMPILER \${GCC_ROOT_DIR}/g++-7)
    set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} -std=c++17")
endif()
          
project(eigen-benchmarks)
          
conan_define_targets()
          
include_directories(\${CONAN_INCLUDE_DIRS_EIGEN})
include_directories(\${CONAN_INCLUDE_DIRS_SPDLOG})
          
set(SOURCE_FILES main.cpp)
add_executable(eigen-benchmarks \${SOURCE_FILES})
install(TARGETS eigen-benchmarks DESTINATION \${CMAKE_CURRENT_BINARY_DIR})`}
          hideLineNumbers
        />
        <br />
        <Typography>
          <Paragraph>The main program is below:</Paragraph>
        </Typography>
        <CppSnippet
          text={`#include <iostream>
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
`}
        />
      </>
    );
  }
}

export default EigenBenchmarks;
