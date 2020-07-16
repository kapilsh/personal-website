import { StandardNormalDistribution } from "./Distributions";
import { sqrt } from "mathjs";

export const BS = (s, k, vol, t, r) => {
  const sigmaSqrtT = vol * Math.sqrt(t);
  const d1 = Math.log(s / k) / sigmaSqrtT + sigmaSqrtT / 2;
  const d2 = d1 - sigmaSqrtT;
  const pvk = k * Math.exp(-r * t);
  const callPrice =
    StandardNormalDistribution.cdf(d1) * s -
    pvk * StandardNormalDistribution.cdf(d2);
  const putPrice =
    StandardNormalDistribution.cdf(-d2) * pvk -
    StandardNormalDistribution.cdf(-d1) * s;
  const callDelta = StandardNormalDistribution.cdf(d1);
  const putDelta = StandardNormalDistribution.cdf(-d1);
  const gamma = StandardNormalDistribution.pdf(d1) / (s * vol * sqrt(t));
  const vega = StandardNormalDistribution.pdf(d1) * s * sqrt(t);
  const thetaPart1 =
    (StandardNormalDistribution.pdf(d1) * s * vol) / (2 * sqrt(t));
  const callTheta = -thetaPart1 - r * pvk * StandardNormalDistribution.cdf(d2);
  const putTheta = -thetaPart1 + r * pvk * StandardNormalDistribution.cdf(-d2);

  return [
    { kind: "price", call: callPrice, put: putPrice },
    { kind: "delta", call: callDelta, put: putDelta },
    { kind: "gamma", call: gamma, put: gamma },
    { kind: "vega", call: vega, put: vega },
    { kind: "theta", call: callTheta, put: putTheta },
  ];
};
