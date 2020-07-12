import React from "react";

import { Typography, Alert } from "antd";
import MathJax from "react-mathjax";

import { PythonSnippet } from "../snippets/PythonSnippet";
import { BashSnippet } from "../snippets/BashSnippet";

import regressionImage from "../../../../static/regression.png";
import graph1 from "../../../../static/geometric_regression_1.png";
import graph2 from "../../../../static/geometric_regression_2.png";

const { Title, Paragraph } = Typography;

class GeometryofRegression extends React.Component {
  render() {
    return (
      <>
        <MathJax.Provider>
          <Typography>
            <Paragraph>
              A picture is worth a thousand words.{" "}
              <a
                href={
                  "http://stats.stackexchange.com/questions/123651/geometric-interpretation-of-multiple-correlation-coefficient-r-and-coefficient"
                }
              >
                This post on stats Stack Exchange
              </a>{" "}
              gives a great description of the geometric representation of
              Linear Regression problems. Let's see this in action using some
              simple examples.
            </Paragraph>
            <Paragraph>
              The below graphic, which appeared in the original stack-exchange
              post, captures the essence of Linear Regression very aptly.
            </Paragraph>
            <img
              alt="regression geometry"
              src={regressionImage}
              style={{
                width: "70%",
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
              }}
            />
            <br />
            <Paragraph>
              The geometrical meaning of the Linear/Multiple Regression fit is
              the projection of predicted variable `y` on{" "}
              <MathJax.Node inline formula={"\\mathbf{span(1, X)}"} /> (with
              constant) or <MathJax.Node inline formula={"\\mathbf{X}"} />{" "}
              (without constant).
            </Paragraph>
            <Paragraph>
              In terms of more generally understood form of Linear Regression:
            </Paragraph>
            <ul>
              <li>
                With Constant:{" "}
                <MathJax.Node
                  inline
                  formula={"\\hat y = \\beta_o + \\beta_1 x"}
                />
              </li>
              <li>
                Without Constant:{" "}
                <MathJax.Node inline formula={"\\hat y = \\beta_1 x"} />
              </li>
            </ul>
            <Paragraph>We will focus on regression with constant.</Paragraph>
            <Paragraph>
              Regression coefficients represent the factors that make a linear
              combination of <MathJax.Node inline formula={"\\mathbb{1}"} /> and{" "}
              <MathJax.Node inline formula={"\\mathbf{X}"} /> i.e. the
              projection of `y` in terms of a linear combination of{" "}
              <MathJax.Node inline formula={"\\mathbb{1}"} /> and{" "}
              <MathJax.Node inline formula={"\\mathbf{X}"} />.
            </Paragraph>
            <Paragraph>
              Additionally, <MathJax.Node inline formula={"\\mathbf{N}"} /> data
              points imply an <MathJax.Node inline formula={"\\mathbf{N}"} />
              -dimensional vector for `y`,{" "}
              <MathJax.Node inline formula={"\\mathbb{1}"} />, and{" "}
              <MathJax.Node inline formula={"\\mathbf{X}"} />. Hence, I will be
              using only three data points for predictor and predicted variables
              to restrict ourselves to 3 dimensions. Reader can create the above
              graphic using the analysis below if they wish.
            </Paragraph>
            <Title level={3}>Analysis</Title>
          </Typography>
          <PythonSnippet
            text={`import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

%matplotlib inline
sp.init_printing(use_unicode=True)
plt.style.use("ggplot")

x = np.array([1.0, 2, 3])
y = np.array([2, 2.5, 5])

Y = sp.Matrix(y)
Y`}
          />
          <MathJax.Node
            formula={
              "\\left[\\begin{matrix}2.0\\\\2.5\\\\5.0\\end{matrix}\\right]"
            }
          />
          <PythonSnippet
            text={`X = sp.Matrix(np.transpose([np.ones(len(x)), x]))\nX`}
          />
          <MathJax.Node
            formula={
              "\\left[\\begin{matrix}1.0 & 1.0\\\\1.0 & 2.0\\\\1.0 & 3.0\\end{matrix}\\right]"
            }
          />
          <PythonSnippet
            text={`fig = plt.figure()
plt.scatter(X.col(1), y)
plt.xlim((0, 5))
plt.ylim((0, 6))
plt.title("Y vs X")
plt.xlabel("X")
plt.ylabel("Y")
plt.gcf().set_size_inches(10, 5)
plt.show()`}
          />
          <img
            alt="graph 1"
            src={graph1}
            style={{
              width: "70%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
          <br />
          <Typography>
            <Title level={3}>Regression Coefficients</Title>
            <Paragraph>
              Linear regression coefficients β are given by:
            </Paragraph>
            <MathJax.Node
              formula={`\\begin{equation}
\\beta = (\\mathbf{X^\\intercal} \\mathbf{X})^{-1} \\mathbf{X^\\intercal} y
\\end{equation}`}
            />
            <Paragraph>
              Let’s calculate β for X and y we defined above.
            </Paragraph>
          </Typography>
          <PythonSnippet
            text={
              "beta = ((X.transpose() * X) ** -1) * X.transpose() * y\nbeta"
            }
          />
          <MathJax.Node
            formula={`\\left[\\begin{matrix}0.166666666666668\\\\1.5\\end{matrix}\\right]`}
          />
          <Typography>
            <Paragraph>
              Since we now have <MathJax.Node inline formula={"\\beta"} />, we
              can calculate the estimated <MathJax.Node inline formula={"y"} />{" "}
              or <MathJax.Node inline formula={"\\hat y"} />.
            </Paragraph>
            <MathJax.Node
              formula={`\\begin{equation}
            \\hat y = \\mathbf{X} \\beta = \\mathbf{X} (\\mathbf{X^\\intercal} \\mathbf{X})^{-1} \\mathbf{X^\\intercal} y
            \\end{equation}`}
            />
          </Typography>
          <PythonSnippet text={"y_hat = X * beta\ny_hat"} />
          <MathJax.Node
            formula={`\\left[\\begin{matrix}1.66666666666667\\\\3.16666666666667\\\\4.66666666666667\\end{matrix}\\right]`}
          />
          <PythonSnippet
            text={`fig = plt.figure()
plt.scatter(x, y)
plt.xlim((0, 5))
plt.ylim((0, 6))
plt.title("Y vs X | Regression Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X.col(1), y_hat, color='blue')
plt.gcf().set_size_inches(10, 5)
plt.show()`}
          />
          <img
            alt="graph 2"
            src={graph2}
            style={{
              width: "70%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
          <br />
          <Typography>
            <Title level={3}>Error Analysis</Title>
            <Paragraph>
              Residuals for the model are given by:{" "}
              <MathJax.Node inline formula={"\\epsilon = \\hat y - y"} />. This
              represents the error in predicted values of `y` using both{" "}
              <MathJax.Node inline formula={"\\mathbb{1}"} /> and{" "}
              <MathJax.Node inline formula={"\\mathbf{X}"} /> in the model. The
              error vector is normal to the{" "}
              <MathJax.Node inline formula={"\\mathbf{span(1, X)}"} /> since it
              represents the component of y that is not in{" "}
              <MathJax.Node inline formula={"\\mathbf{span(1, X)}"} />.
            </Paragraph>
          </Typography>
          <PythonSnippet text={"res = y - y_hat\nres"} />
          <MathJax.Node
            formula={`\\left[\\begin{matrix}0.333333333333332\\\\-0.666666666666668\\\\0.333333333333332\\end{matrix}\\right]`}
          />
          <Typography>
            <Paragraph>
              Average vector or <MathJax.Node inline formula={"\\bar y"} /> is
              geometrically the projection of `y` on just the{" "}
              <MathJax.Node inline formula={"\\mathbb{1}"} /> vector.
            </Paragraph>
          </Typography>
          <PythonSnippet
            text={"y_bar = np.mean(y) * sp.Matrix(np.ones(len(y)))\ny_bar"}
          />
          <MathJax.Node
            formula={`\\left[\\begin{matrix}3.16666666666667\\\\3.16666666666667\\\\3.16666666666667\\end{matrix}\\right]`}
          />
          <Typography>
            <Paragraph>
              We can calculate the error in the average model or where we
              represent the predicted values as the average vector{" "}
              <MathJax.Node inline formula={"\\bar y"} />. Error in the model is
              given by <MathJax.Node inline formula={"\\kappa = \\bar y - y"} />
              .
            </Paragraph>
          </Typography>
          <PythonSnippet text={"kappa = y_bar - y\nkappa"} />
          <MathJax.Node
            formula={`\\left[\\begin{matrix}1.16666666666667\\\\0.666666666666667\\\\-1.83333333333333\\end{matrix}\\right]`}
          />
          <Typography>
            <Paragraph>
              Both <MathJax.Node inline formula={"\\bar y"} /> and{" "}
              <MathJax.Node inline formula={"\\hat y"} /> are predictors for `y`
              and it is reasonable to calculate how much error we reduce by
              adding <MathJax.Node inline formula={"\\mathbf{X}"} /> to the
              model. Let's call the error{" "}
              <MathJax.Node inline formula={"\\eta"} />.
            </Paragraph>
          </Typography>
          <PythonSnippet text={"eta = y_hat - y_bar\neta"} />
          <MathJax.Node
            formula={`\\left[\\begin{matrix}-1.5\\\\1.77635683940025 \\cdot 10^{-15}\\\\1.5\\end{matrix}\\right]`}
          />
          <Typography>
            <Paragraph>
              Now from here we can prove whether{" "}
              <MathJax.Node inline formula={"\\eta"} /> and{" "}
              <MathJax.Node inline formula={"\\epsilon"} /> are perpendicular to
              each other. We can check it by calculating their dot product.
            </Paragraph>
          </Typography>
          <MathJax.Node
            formula={`\\left[\\begin{matrix}5.55111512312578 \\cdot 10^{-16}\\end{matrix}\\right]`}
          />
          <Alert
            message="NOTE"
            description={
              <p>
                Hence, we can see that <MathJax.Node inline formula={"\\eta"} />{" "}
                and <MathJax.Node inline formula={"\\epsilon"} /> are normal to
                each other since their dot product is 0
              </p>
            }
            type="info"
            showIcon
          />
          <br />
          <Typography>
            <Paragraph>
              From here we can also prove the relationship between Total Sum of
              Squares (SST), Sum of Squares due to Squares of Regression (SSR)
              and Sum of Squares due to Squares of Errors (SSE)
            </Paragraph>
            <MathJax.Node
              formula={`\\mathbf{SST} = \\mathbf{SSR} + \\mathbf{SSE}`}
            />
            <ul>
              <li>
                <MathJax.Node inline formula={"\\mathbf{SST}"} /> can be
                represented by the squared norm of{" "}
                <MathJax.Node inline formula={"\\kappa"} />
              </li>
              <li>
                <MathJax.Node inline formula={"\\mathbf{SSR}"} /> can be
                represented by the squared norm of{" "}
                <MathJax.Node inline formula={"\\eta"} />
              </li>
              <li>
                <MathJax.Node inline formula={"\\mathbf{SSE}"} /> can be
                represented by the squared norm of{" "}
                <MathJax.Node inline formula={"\\epsilon"} />
              </li>
            </ul>
            <Paragraph>
              We can use Pythagorean Theorem to check the above relationship
              i.e.
            </Paragraph>
            <MathJax.Node
              formula={`\\begin{equation}
              ||\\kappa||^2 = ||\\eta||^2 + ||\\epsilon||^2
              \\end{equation}`}
            />
          </Typography>
          <PythonSnippet
            text={"kappa.norm() ** 2  - eta.norm() ** 2 - res.norm() ** 2"}
          />
          <BashSnippet text={"0.0"} hideLineNumbers />
          <br />
          <Alert
            message="NOTE"
            description={
              <p>
                Hence, as we expected, κ, η and ϵ form a right angled triangle.
              </p>
            }
            type="info"
            showIcon
          />
          <br />
          <Typography>
            <Title level={3}>Summary</Title>
            <Paragraph>
              Through this post, I demonstrated how we can interpret
              linear/multiple regression geometrically. Also, I solved a linear
              regression model using Linear Algebra.
            </Paragraph>
          </Typography>
        </MathJax.Provider>
      </>
    );
  }
}

export default GeometryofRegression;
