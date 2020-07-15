import { erf, sqrt } from "mathjs";

const NormalDistribution = {
  cdf: (x, mean, sigma) => {
    return (1 - erf((mean - x) / (sqrt(2) * sigma))) / 2;
  },
  pdf: (x, mean, sigma) => {
    const normalized = (x - mean) / sigma;
    return (
      Math.exp(-0.5 * normalized * normalized) / (sigma * sqrt(2 * Math.PI))
    );
  },
};

const StandardNormalDistribution = {
  cdf: (x) => {
    return NormalDistribution.cdf(x, 0, 1.0);
  },
  pdf: (x) => {
    return NormalDistribution.pdf(x, 0, 1.0);
  },
};

export { NormalDistribution, StandardNormalDistribution };
