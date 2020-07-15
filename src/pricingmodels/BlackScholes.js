import { StandardNormalDistribution } from "./Distributions";

export const BS = (s, k, vol, t, r) => {
  const sigmaSqrtT = vol * Math.sqrt(t);
  const d1 = Math.log(s / k) / sigmaSqrtT + sigmaSqrtT / 2;
  const d2 = d1 - sigmaSqrtT;
  const pvk = k * Math.exp(-r * t);
  return StandardNormalDistribution.cdf(d1) * s - pvk * StandardNormalDistribution.cdf(d2);
};
