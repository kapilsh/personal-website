import React from "react";
import { Typography, Alert } from "antd";
import MathJax from "react-mathjax";

import { PythonSnippet } from "../snippets/PythonSnippet";
import { BashSnippet } from "../snippets/BashSnippet";

const { Title, Paragraph } = Typography;

class GramSchmidtProcedure extends React.Component {
  render() {
    return (
      <>
        <Typography>
          <MathJax.Provider>
            <Paragraph>
              An interesting way to understand Linear Regression is{" "}
              <a
                href={
                  "https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process"
                }
              >
                Gram-Schmidt Method
              </a>{" "}
              of successive projections to calculate the coefficients of
              regression. Gram-Schmidt procedure transforms the variables into a
              new set of orthogonal or uncorrelated variables. On applying the
              procedure, we should get exactly the same regression coefficients
              as with projection of predicted variable on the feature space.
            </Paragraph>
            <Paragraph>
              Linear Model with number of inputs (p) such{" "}
              <MathJax.Node inline formula={"p > 1"} /> is called multiple
              regression. We can represent least squares estimates of multiple
              regression in terms of estimates of univariate linear model. To
              understand this, let us assume a multivariate{" "}
              <MathJax.Node inline formula={"(p > 1)"} /> linear model:
              <MathJax.Node
                inline
                formula={"\\mathbf{Y} = \\mathbf{X}\\beta + \\epsilon"}
              />
              .
            </Paragraph>
            <Paragraph>The least square estimate and residuals are:</Paragraph>
            <MathJax.Node
              formula={`\\begin{equation}
\\hat \\beta = \\dfrac{\\sum_{n=1}^{N} x_i y_i}{\\sum_{n=1}^{N} x_i^2 }
\\end{equation}`}
            />
            <Paragraph>and</Paragraph>
            <MathJax.Node
              formula={`\\begin{equation}
              r_i = y_i - x_i \\hat \\beta
              \\end{equation}`}
            />
            <Paragraph>
              In convenient vector notation, we let{" "}
              <MathJax.Node
                inline
                formula={"\\mathbf{y} = (y_1, ..., y_N)^\\intercal"}
              />
              ,{" "}
              <MathJax.Node
                inline
                formula={"\\mathbf{x} = (x_1, ..., x_N)^\\intercal"}
              />{" "}
              and define:
              <MathJax.Node
                formula={`\\begin{equation}
            \\langle \\mathbf{x}, \\mathbf{y} \\rangle = \\sum_{n=1}^{N} x_i y_i = \\mathbf{x}^\\intercal \\mathbf{y}
            \\end{equation}`}
              />
              Hence, we can write the parameters in terms of inner product of x
              and y.
              <MathJax.Node
                formula={`\\begin{equation}
              \\hat \\beta = \\dfrac{\\langle \\mathbf{x}, \\mathbf{y} \\rangle}{\\langle \\mathbf{x}, \\mathbf{x} \\rangle};
              \\mathbf{r} = \\mathbf{y} - \\mathbf{x} \\hat \\beta
              \\end{equation}`}
              />
              <Alert
                message="NOTE"
                description="The inner product notation generalizes the linear regression to different metric spaces, as well as to probability spaces."
                type="info"
                showIcon
              />
              <br />
            </Paragraph>
            <Paragraph>
              If the inputs `x_1`, `x_2`, ..., `x_p` are orthogonal, i.e.{" "}
              <MathJax.Node
                inline
                formula={"\\langle \\mathbf{x_j}, \\mathbf{x_k} \\rangle = 0"}
              />{" "}
              for all <MathJax.Node inline formula={"j \\neq k"} />, then it is
              easy to check that the multiple least squares estimates{" "}
              <MathJax.Node inline formula={"\\beta_j"} /> are equal to{" "}
              <MathJax.Node
                inline
                formula={
                  "\\langle \\mathbf {x_j}, \\mathbf{y} \\rangle / \\langle \\mathbf{x_j}, \\mathbf{x_j} \\rangle"
                }
              />
              .
            </Paragraph>
            <Paragraph>
              Now if we have an intercept and a single input x, we can find that
            </Paragraph>
            <MathJax.Node
              formula={`\\begin{equation}
                \\hat \\beta_1 = \\dfrac{\\langle \\mathbf{x} - \\bar x \\mathbb{1}, \\mathbf{y} \\rangle}{\\langle \\mathbf{x} - \\bar x \\mathbb{1}, \\mathbf{x} - \\bar x \\mathbb{1} \\rangle}
                \\end{equation}`}
            />
            <Paragraph>
              where{" "}
              <MathJax.Node
                inline
                formula={"\\bar x = \\sum_{(n = 1)}^{N} x_i / N"}
              />{" "}
              and{" "}
              <MathJax.Node inline formula={"\\mathbb{1} = \\mathbf{x}_0"} />,
              the vector of N ones.
            </Paragraph>
            <Paragraph>
              The steps to generate the regression using this procedure -
              <ol>
                <li>
                  Regress x on `1` to produce the residual{" "}
                  <MathJax.Node
                    inline
                    formula={"\\mathbf{z} = \\mathbf{x} - \\bar x \\mathbb{1}"}
                  />
                </li>
                <li>
                  Regress <MathJax.Node inline formula={"\\mathbf{y}"} /> on the
                  residual `z` to give the coefficient{" "}
                  <MathJax.Node inline formula={"\\hat \\beta_1"} />
                </li>
              </ol>
              where, "regress `b` on `a`" means a single univariate regression
              of `b` on `a` with no intercept.
            </Paragraph>
            <Paragraph>
              This process also generalizes to p points and is called Gram -
              Schmidt Process. It can be understood as a process of successive
              orthogonalization of the inputs, starting from `1`.
            </Paragraph>
            <Title level={3}>Gram-Schmidt Process Algorithm</Title>
            <ol>
              <li>Initialize `z_0` = `x_0` = `1`</li>
              <li>
                For all <MathJax.Node inline formula={"\\mathbf{x_j}"} /> s.t.
                `j` in <MathJax.Node inline formula={"\\{1, 2, 3, ..., p\\}"} />{" "}
                for `p` inputs, regress `x_j` on the residuals after{" "}
                <MathJax.Node inline formula={"j_{th}"} /> step, where the
                coefficients{" "}
                <MathJax.Node inline formula={"\\hat \\gamma_{lj}"} /> are:
                <MathJax.Node
                  formula={`\\begin{equation}
                  \\hat \\gamma_{lj} = \\dfrac{\\langle \\mathbf{z_l}, \\mathbf{x_j} \\rangle}{\\langle \\mathbf{z_l}, \\mathbf{z_l} \\rangle}
                  \\end{equation}`}
                />
              </li>
              <li>
                and residuals at each step are:
                <MathJax.Node
                  formula={`\\begin{equation}
\\mathbf{z_j} = \\mathbf{x_j} - \\sum_{k=0}^{j-1} \\hat \\gamma_{kj}\\mathbf{z_k}
\\end{equation}`}
                />
              </li>
              <li>
                Finally, we can calculate{" "}
                <MathJax.Node inline formula={"\\hat \\beta_p"} /> as:
                <MathJax.Node
                  formula={`\\begin{equation}
                  \\hat \\beta_p = \\dfrac{\\langle \\mathbf{z_p}, \\mathbf{y} \\rangle}{\\langle \\mathbf{z_p}, \\mathbf{z_p} \\rangle}
                  \\end{equation}`}
                />
              </li>
            </ol>
            <Paragraph>Let us test this procedure with p = 2.</Paragraph>
            <Paragraph>
              As a first step, we will run a Multiple Regression over a set of
              inputs and get the regression coefficients.
            </Paragraph>
          </MathJax.Provider>
        </Typography>
        <PythonSnippet
          text={`import numpy as np
from sklearn.linear_model import LinearRegression

x1 = np.array([2, 2.2, 3.2, 4.5, 5.0])
x2 = np.array([45.0, 20.0, 30.0, 10.0, 6.5])
x0 = np.ones(len(x1))
z0 = x0
y = np.array([2.3, 4.5, 6.7, 8.9, 10.11])

X = np.matrix([x0, x1, x2]).T
Y = np.matrix(y).T

lin_reg = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
lin_reg.fit(X, Y)
coeffs = lin_reg.coef_
print("Coefficients: {}".format(coeffs))`}
        />
        <BashSnippet
          text={`Coefficients: [[ 1.39782326  1.83576285 -0.04935882]]`}
          hideLineNumbers
        />
        <br />
        <Typography>
          <MathJax.Provider>
            <Paragraph>
              Now, let us calculate the{" "}
              <MathJax.Node inline formula={"\\beta_2"} /> using the iterative
              procedure.
            </Paragraph>
          </MathJax.Provider>
        </Typography>
        <PythonSnippet
          text={`# Step 1
gamma_01 = z0.dot(x1) / (z0.dot(z0))
z1 = x1 - gamma_01 * z0

# Step 2
gamma_02 = z0.dot(x2) / (z0.dot(z0))
gamma_12 = z1.dot(x2) / (z1.dot(z1))
z2 = x2 - gamma_02 * z0 - gamma_12 * z1

# Step 3
beta_p = z2.dot(y) / (z2.dot(z2))
print(beta_p)`}
        />
        <BashSnippet text={`-0.0493588203647`} hideLineNumbers />
        <br />
        <Typography>
          <MathJax.Provider>
            <Paragraph>
              Similarly, we can calculate the{" "}
              <MathJax.Node inline formula={"\\beta_1"} /> using the iterative
              procedure.
            </Paragraph>
          </MathJax.Provider>
        </Typography>
        <PythonSnippet
          text={`gamma_01 = z0.dot(x2) / (z0.dot(z0))
z1 = x2 - gamma_01 * z0
          
gamma_02 = z0.dot(x1) / (z0.dot(z0))
gamma_12 = z1.dot(x1) / (z1.dot(z1))
z2 = x1 - gamma_02 * z0 - gamma_12 * z1
          
beta_p = z2.dot(y) / (z2.dot(z2))
print(beta_p)`}
        />
        <BashSnippet text={`1.83576285118`} hideLineNumbers />
        <br />
        <Typography>
          <MathJax.Provider>
            <Paragraph>
              As we can see that both{" "}
              <MathJax.Node inline formula={"\\beta_1"} /> and{" "}
              <MathJax.Node inline formula={"\\beta_2"} /> match the regression
              coefficients obtained via L2 - norm minimization.
            </Paragraph>
            <Paragraph>
              We can represent the transformations on `Z` more generally as:
              <MathJax.Node
                formula={`\\begin{equation}
                  \\mathbf{X} = \\mathbf{Z} \\mathbf{\\Gamma}
                  \\end{equation}`}
              />
              where `z_j` are the columns of `Z` and{" "}
              <MathJax.Node inline formula={"\\Gamma"} /> is an upper triangular
              matrix with the coefficients{" "}
              <MathJax.Node inline formula={"\\gamma_{lj}"} />.
            </Paragraph>
            <Paragraph>
              For p=2 case, we have:
              <MathJax.Node
                formula={`\\begin{equation}
                \\mathbf{\\Gamma} =
                  \\begin{bmatrix}
                    1 & \\hat \\gamma_{01} & \\hat \\gamma_{02} \\\\
                    0 & 1 & \\hat \\gamma_{12} \\\\
                    0 & 0 & 1
                  \\end{bmatrix}
                \\end{equation}`}
              />
            </Paragraph>
            <Paragraph>
              This is similar to QR Decomposition. We can do a scaled
              QR-decomposition as:
              <MathJax.Node
                formula={`\\begin{equation}
\\mathbf{X} = \\mathbf{Z} \\mathbf{D^{-1}} \\mathbf{D} \\mathbf{\\Gamma} = \\mathbf{Q} \\mathbf{R}
\\end{equation}`}
              />
              where,
              <MathJax.Node
                formula={`D_{jj} = ||\\mathbf{z_j}||, and \\\\
                \\mathbf{Q^\\intercal} = \\mathbf{Q^{-1}}`}
              />
              From here, we can calculate{" "}
              <MathJax.Node inline formula={"\\hat \\beta"} />:
            </Paragraph>
            <Paragraph>
              <MathJax.Node
                formula={`\\begin{equation}
\\hat \\beta = (\\mathbf{X^\\intercal} \\mathbf{X})^{-1} \\mathbf{X^\\intercal} y \\\\
    = (\\mathbf{(QR)^\\intercal} \\mathbf{QR})^{-1} \\mathbf{(QR)^\\intercal} y \\\\
    = (\\mathbf{R^\\intercal} \\mathbf{Q^\\intercal} \\mathbf{Q} \\mathbf{R})^{-1} \\mathbf{R^\\intercal} \\mathbf{Q^\\intercal} y \\\\
    = (\\mathbf{R^\\intercal} \\mathbf{I} \\mathbf{R})^{-1} \\mathbf{R^\\intercal} \\mathbf{Q^\\intercal} y \\\\
    = (\\mathbf{R^\\intercal} \\mathbf{R})^{-1} \\mathbf{R^\\intercal} \\mathbf{Q^\\intercal} y \\\\
    = \\mathbf{R^{-1}} (\\mathbf{R^\\intercal})^{-1} \\mathbf{R^\\intercal} \\mathbf{Q^\\intercal} y \\\\
    = \\mathbf{R^{-1}}  \\mathbf{Q^\\intercal} y \\\\
\\end{equation}`}
              />
              and,
              <MathJax.Node
                formula={`\\begin{equation}
                \\hat {\\mathbf{y}} = \\mathbf{X} \\hat \\beta \\\\
                    = \\mathbf{Q} \\mathbf{R} \\mathbf{R^{-1}} \\mathbf{Q^\\intercal} y \\\\
                    = \\mathbf{Q} \\mathbf{Q^\\intercal} y \\\\
                \\end{equation}`}
              />
            </Paragraph>
            <Title level={3}>Sources</Title>
            <Paragraph>
              <ol>
                <li>
                  <a href="https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576">
                    Elements of Statistical Learning
                  </a>
                </li>
              </ol>
            </Paragraph>
          </MathJax.Provider>
        </Typography>
      </>
    );
  }
}

export default GramSchmidtProcedure;
