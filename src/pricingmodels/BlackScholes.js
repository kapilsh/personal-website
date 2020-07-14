import { NormalDistribution } from "./Distributions";

export const BS = (s, f, vol, t, r) => {
  const sigmaSqrtT = vol * Math.sqrt(t);
  const d1 = Math.log(s / k) / sigmaSqrtT + sigmaSqrtT / 2;
  const d2 = d1 - sigmaSqrtT;
  const pvk = k * Math.exp(-r * t);
  return NormalDistribution.cdf(d1) * s - pvk * NormalDistribution.cdf(d2);
};
