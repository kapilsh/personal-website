import { erf, sqrt } from 'mathjs'


const NormalDistribution = {
  cdf: (x, mean, sigma) => {
    var z = (x - mean) / Math.sqrt(2 * sigma * sigma);
    var t = 1 / (1 + 0.3275911 * Math.abs(z));
    var a1 = 0.254829592;
    var a2 = -0.284496736;
    var a3 = 1.421413741;
    var a4 = -1.453152027;
    var a5 = 1.061405429;
    var erf = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
    var sign = 1;
    if (z < 0) {
      sign = -1;
    }
    return (1 / 2) * (1 + sign * erf);
  },
};

const StandardNormalDistribution = {
  cdf: (x) => {
    return NormalDistribution.cdf(x, 0, 1.0);
  },
};

export { NormalDistribution, StandardNormalDistribution };
