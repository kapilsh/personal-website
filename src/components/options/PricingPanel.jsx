import React from "react";
import { PageHeader, Row, Col, InputNumber, Form } from "antd";
import { useLocalStore, useObserver } from "mobx-react"

import { BS } from "../../pricingmodels/BlackScholes";
import { exp } from "mathjs";

const StoreContext = React.createContext();

const StoreProvider = ({ children }) => {
  const store = useLocalStore(() => ({
    strike: 100.0,
    underlying: 100.0,
    volatility: 0.16,
    time: 0.25,
    rate: 0.00,
    setStrike: x => (store.strike = x),
    setUnderlying: x => (store.underlying = x),
    setVolatility: x => (store.volatility = x),
    setTime: x => (store.time = x),
    setRate: x => (store.rate = x),
    get price() {
      console.log("Recompute")
      return BS(store.underlying, store.strike, store.volatility, store.time, store.rate)
    }
  }))
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>
}

const Parameter = (props) => {
  return (<Row>
    <Col span={16}>
      <Form.Item label={props.label} />
    </Col>
    <Col span={8}>
      <InputNumber
        step={props.step}
        value={props.value}
        onChange={(value) => {
          console.log(`${props.label} changed to ${value}`);
          props.setValue(value)
        }}
      />
    </Col>
  </Row>)
}

const Parameters = () => {
  const store = React.useContext(StoreContext)
  // const [strike, setStrike] = useState(props.strike);
  // const underlyingPrice = props.underlying;
  // const volatility = props.volatility;
  // const riskFreeRate = props.riskFreeRate;
  // const t = props.time;
  // const recalculate = props.recalculate;

  return useObserver(() => (
    <>
      <Parameter value={store.strike} step={0.25} setValue={store.setStrike} label={"Strike"} />
      <Parameter value={store.underlying} step={0.25} setValue={store.setUnderlying} label={"Underlying Price"} />
      <Parameter value={store.volatility} step={0.0001} setValue={store.setVolatility} label={"Volatility"} />
      <Parameter value={store.rate} step={0.0001} setValue={store.setRate} label={"Rate"} />
      <Parameter value={store.time} step={0.0001} setValue={store.setTime} label={"Time to Expiry"} />
      <div>
        {store.price}
      </div>
    </>
  ))
};


const PricingPanel = (props) => {
  return (
    <StoreProvider>
      <PageHeader
        className="site-page-header"
        title="Options Playground"
        subTitle="Playground to play with options pricing models"
      />
      <Parameters />
    </StoreProvider>
  )
}

export default PricingPanel;

// class Options extends React.Component {
//   render() {
//     return (
//       <>

//         <Row>
//           <Col span={12}>
//             {/* <ParametersForm strike={100.0} underlying={100.0} /> */}
//             <Parameters strike={100.0} recalculate={(p) => console.log(p)} />
//           </Col>
//           <Col></Col>
//         </Row>
//       </>
//     );
//   }
// }
