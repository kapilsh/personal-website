const mathjs = import("mathjs");

const NormalDistribution = {
  cdf: (x, mean, standardDeviation) => {
    return (
      (1 - mathjs.erf((mean - x) / (Math.sqrt(2) * standardDeviation))) / 2
    );
  },
};

const StandardNormalDistribution = {
  cdf: (x) => {
    return NormalDistribution.cdf(x, 0, 1.0);
  },
};

export { NormalDistribution, StandardNormalDistribution };
