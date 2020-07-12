import React from "react";
import { Typography, Alert } from "antd";
import MathJax from "react-mathjax";
import { CodeBlock, dracula } from "react-code-blocks";
import ReactPlayer from "react-player";

import Image1 from "../../../../static/local_linear_regression_bokeh_plot_1.png";
import Image2 from "../../../../static/local_linear_regression_bokeh_plot_2.png";
import Image3 from "../../../../static/local_linear_regression_bokeh_plot_3.png";
import Image4 from "../../../../static/local_linear_regression_bokeh_plot_4.png";
import Image5 from "../../../../static/local_linear_regression_bokeh_plot_5.png";
import Image6 from "../../../../static/local_linear_regression_bokeh_plot_6.png";
import Image7 from "../../../../static/local_linear_regression_bokeh_plot_7.png";

const { Title, Paragraph } = Typography;

const PythonSnippet = (props) => {
  return (
    <div
      style={{
        fontFamily: "Source Code Pro",
      }}
    >
      <CodeBlock
        text={props.snippet}
        language={"python"}
        showLineNumbers={!props.hideLines}
        theme={dracula}
        wrapLines
      />
      <br />
    </div>
  );
};

const predictFunctionPy = `def predict(x_test, x_train, y_train, h):
    if len(x_train) != len(y_train):
        raise ValueError("X and Y Should have same length")
    B = np.array([np.ones(len(x_train)), x_train]).T
    y_hat = []
    for x0 in x_test:
        W = np.diag(gaussian_kernel(x_train , x0, h))
        y_hat.append(np.array([1, x0]).T.dot(
            np.linalg.inv(B.T.dot(W).dot(B))).dot(
            B.T).dot(W).dot(y_train))

    return np.array(y_hat)`;

const bandwidthComparisonPy = `h_values = [0.01, 0.1, 1]
colors = ["#A6E22E", "#FD971F", "#AE81FF"]

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
show(p)`;

const linearLinesPy = `h_trial = 0.1
x_trials = np.linspace(domain[0], domain[1], 6)

def local_coeffs(x_0, x_train, y_train, h):
    if len(x_train) != len(y_train):
        raise ValueError("X and Y Should have same length")
    
    B = np.array([np.ones(len(x_train)), x_train]).T
    W = np.diag(gaussian_kernel(x_train , x_0, h_trial))
    return np.linalg.inv(B.T.dot(W).dot(B)).dot(B.T).dot(W).dot(y_train)

coeffs = [local_coeffs(x_0, x, y, h_trial) for x_0 in x_trials]
print(coeffs)`;

const linearLinesOutput = `[array([ 1.10375711, -0.24977984]), 
array([ 1.93831019,  3.12814681]), 
array([ 2.04711505,  3.62427698]), 
array([ 2.58841427,  1.23608707]), 
array([ 3.96096982, -2.14935988]), 
array([ 5.16166187, -4.00951684])]`;

const plotLinearLinesPy = `p = figure(plot_width=800, plot_height=400)
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
show(p)`;

class LocalLinearRegression extends React.Component {
  render() {
    return (
      <>
        <Typography>
          <Paragraph>
            I previously wrote a{" "}
            <a href="http://www.sharmakapil.com/2018/08/26/kernel-smoothing.html">
              post
            </a>{" "}
            about <strong>Kernel Smoothing</strong> and how it can be used to
            fit a non-linear function non-parametrically. In this post, I will
            extend on that idea and try to mitigate the disadvantages of kernel
            smoothing using <strong>Local Linear Regression</strong>.
          </Paragraph>
          <Title level={3}>Setup</Title>
          <Paragraph>
            <MathJax.Provider>
              I generated some data in my previous{" "}
              <a href="http://www.sharmakapil.com/2018/08/26/kernel-smoothing.html">
                post
              </a>{" "}
              and I will reuse the same data for this post. The data was
              generated from the function{" "}
              <MathJax.Node
                inline
                formula={"\\mathbf{y = f(x) = sin(4x) + 2}"}
              />{" "}
              with some <strong>Gaussian</strong> noise and here's how it looks:
            </MathJax.Provider>
          </Paragraph>
        </Typography>
        <img
          alt="scatter plot"
          src={Image1}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <Typography>
          <Title level={3}>Local Linear Regression</Title>
          <Paragraph>
            As I mentioned in the previous article, in kernel smoothing
            out-of-sample predictions on the edges and in sparse regions can
            have significant errors and bias. In{" "}
            <strong>Local Linear Regression</strong>, we try to reduce this bias
            to first order, by fitting straight lines instead of local
            constants.
            <MathJax.Provider>
              Local linear regression solves a weighted least squares problem at
              each out-of-sample point <MathJax.Node inline formula={"x_0"} />,
              given by:
              <MathJax.Node
                formula={`\\begin{equation} 
                \\mathbf{\\hat y_0 = \\hat \\alpha(x_0) + \\hat \\beta(x_0) x_0}
                \\end{equation}`}
              />
              <Alert
                message="NOTE"
                description="Even though we fit an entire linear model, we only use it to fit a single point."
                type="info"
                showIcon
              />
              <br />
              Let's formulate the matrix expression to calculate{" "}
              <MathJax.Node inline formula={"\\hat y_0"} /> and then implement
              it in Python. Let:
              <ul>
                <li>
                  <MathJax.Node inline formula={"b(x)^T"} /> be a 2-d vector
                  given by:{" "}
                  <MathJax.Node inline formula={"b(x)^T = (1, x_0)"} />
                </li>
                <li>
                  <MathJax.Node inline formula={"\\mathbf{B}"} /> be a{" "}
                  <MathJax.Node inline formula={"N \\times 2"} /> matrix with
                  the <MathJax.Node inline formula={"i^{th}"} /> row{" "}
                  <MathJax.Node inline formula={"b(x)^T"} />
                </li>
                <li>
                  <MathJax.Node inline formula={"\\mathbf{W(x_0)}"} /> be{" "}
                  <MathJax.Node inline formula={"N \\times N"} /> diagonal
                  matrix with <MathJax.Node inline formula={"i^{th}"} />{" "}
                  diagonal element{" "}
                  <MathJax.Node inline formula={"K_\\lambda(x_0, x_i)"} />
                </li>
              </ul>
              <Paragraph>Then,</Paragraph>
              <MathJax.Node
                formula={`\\begin{equation} 
              \\mathbf{\\hat y_0} = b(x_0)^\\intercal (\\mathbf{B}^\\intercal\\mathbf{W}\\mathbf{B})^{-1} \\mathbf{B}^\\intercal \\mathbf{W(x_0)} \\mathbf{y} 
              \\end{equation}`}
              />
              <Alert
                message="NOTE"
                description={
                  <ul>
                    <li>
                      The final estimate{" "}
                      <MathJax.Node inline formula={"\\hat y_0"} /> is still
                      linear in <MathJax.Node inline formula={"y_i's"} /> since
                      the weights do not depend on `y` at all
                    </li>
                    <li>
                      In other words,{" "}
                      <MathJax.Node
                        inline
                        formula={
                          "b(x_0)^\\intercal (\\mathbf{B}^\\intercal\\mathbf{W}\\mathbf{B})^{-1} \\mathbf{B}^\\intercal\\mathbf{W(x_0)}"
                        }
                      />{" "}
                      is a linear operator on y and is independent of y
                    </li>
                  </ul>
                }
                type="info"
                showIcon
              />
            </MathJax.Provider>
          </Paragraph>
        </Typography>
        <PythonSnippet snippet={predictFunctionPy} />
        <Typography>
          <Paragraph>
            Let's choose a few bandwidth values and check the fits:
          </Paragraph>
        </Typography>
        <PythonSnippet snippet={bandwidthComparisonPy} />
        <img
          alt="bandwidth comparison"
          src={Image2}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <Typography>
          <Paragraph>
            <MathJax.Provider>
              To illustrate how the algorithm works, I wil choose a few x values
              and show the local linear fits for each of those points. I will
              use h = 0.1 since the corresponding fit looks pretty reasonable.
              As explained above, we will get the corresponding α(x0) and β(x0)
              for each point.
            </MathJax.Provider>
          </Paragraph>
        </Typography>
        <PythonSnippet snippet={linearLinesPy} />
        <PythonSnippet snippet={linearLinesOutput} hideLines />
        <Typography>
          <Paragraph>
            Now that we have the local coefficients, let's plot the local lines
            at each point in x_trial and the complete fit.
          </Paragraph>
        </Typography>
        <PythonSnippet snippet={plotLinearLinesPy} hideLines />
        <img
          alt="linear lines"
          src={Image3}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <Typography>
          <Paragraph>
            One great resource that I came across related to local linear
            regression is the lecture below (
            <strong>
              Statistical Machine Learning, Larry Wasserman, CMU, Jan 2016
            </strong>
            ):
          </Paragraph>
          <ReactPlayer
            url="https://youtu.be/e9mN6UH5QIQ"
            style={{
              width: "70%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
          <br />
          <Paragraph>
            As in the previous post, I will end this post by estimating optimal
            bandwidth using <strong>Leave One Out Cross Validation</strong> and{" "}
            <strong>K-Fold Cross Validation</strong> below:
          </Paragraph>
          <Title level={3}>Cross Validation</Title>
          <Title level={4}>Leave One Out Cross Validation (LOOCV)</Title>
        </Typography>
        <PythonSnippet snippet={"h_range = np.linspace(0.01, 0.2, 20)"} />
        <img
          alt="loocv cross validation"
          src={Image4}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <PythonSnippet snippet={"h_optimal : 0.09"} hideLines />
        <img
          alt="loocv optimal fit"
          src={Image5}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <Typography>
          <Title level={4}>K-Fold Cross Validation</Title>
        </Typography>
        <PythonSnippet snippet={`num_folds = 10\nnum_tries = 5`} />
        <PythonSnippet snippet={"h_optimal : 0.06"} hideLines />
        <img
          alt="scatter plot"
          src={Image6}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <img
          alt="scatter plot"
          src={Image7}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
        <br />
        <Typography>
          <Title level={3}>Final Words</Title>
          <Paragraph>
            In this post, we extended the <strong>Kernel Smoothing</strong>{" "}
            technique to fit local linear functions instead of local constants
            at each input point. Fitting locally linear functions helps us
            reduce the bias and error on the edges and sparse regions of our
            data.
          </Paragraph>
          <Title level={3}>Sources</Title>
          <Paragraph>
            <ol>
              <li>
                <a href="https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576">
                  Elements of Statistical Learning - Chapter 6
                </a>
              </li>
              <li>
                <a href="https://www.youtube.com/watch?v=e9mN6UH5QIQ">
                  Non-Parametric regression
                </a>
              </li>
            </ol>
          </Paragraph>
        </Typography>
      </>
    );
  }
}

export default LocalLinearRegression;
