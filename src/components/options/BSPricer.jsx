import React from "react";
import { PageHeader, Row, Col, InputNumber, Form, Table, Tabs, Switch, Alert } from "antd";
import { useLocalStore, useObserver } from "mobx-react";

import { BS } from "../../pricingmodels/BlackScholes";

const { ErrorBoundary } = Alert;

const StoreContext = React.createContext();

const StoreProvider = ({ children }) => {
  const store = useLocalStore(() => ({
    strike: 100.0,
    underlying: 100.0,
    volatility: 0.16,
    time: 0.25,
    rate: 0.0,
    solverPrice: 3.20,
    solverType: "Call",
    useSolver: false,
    setStrike: (x) => (store.strike = x),
    setUnderlying: (x) => (store.underlying = x),
    setVolatility: (x) => (store.volatility = x),
    setTime: (x) => (store.time = x),
    setRate: (x) => (store.rate = x),
    setUseSolver: (x) => (store.useSolver = x),
    setSolverPrice: (x) => (store.solverPrice = x),
    setSolverType: (x) => (store.solverType = x),
    get bs() {
      return BS(
        store.underlying,
        store.strike,
        store.volatility,
        store.time,
        store.rate
      );
    },
    get extrinsicPrices() {
      const strikes = [...Array(100).keys()].map((x) => (2 * x) / 100.0);
      return strikes.map((strike) => {
        const bs = BS(
          store.underlying,
          strike,
          store.volatility,
          0,
          store.rate
        );
        return { Call: bs.Call.price, Put: bs.Put.price };
      });
    },
    get intrinsicPrices() {
      const strikes = [...Array(100).keys()].map((x) => (2 * x) / 100.0);
      return strikes.map((strike) => {
        return {
          Call: Math.max(strike - state.underlying, 0),
          Put: Math.max(state.underlying - strike, 0),
        };
      });
    },
  }));
  return (
    <StoreContext.Provider value={store}>{children}</StoreContext.Provider>
  );
};

const Parameter = (props) => {
  return (
    <Row>
      <Col span={16}>
        <Form.Item label={props.label} />
      </Col>
      <Col span={8}>
        <InputNumber
          step={props.step}
          value={props.value}
          onChange={(value) => {
            console.log(`${props.label} changed to ${value}`);
            props.setValue(value);
          }}
        />
      </Col>
    </Row>
  );
};

const ThrowError = () => {


  const onClick = () => {
    setError(new Error('An Uncaught Error'));
  };



  return (
    <Button type="danger" onClick={onClick}>
      Click me to throw a error
    </Button>
  );
};


const Parameters = () => {
  const store = React.useContext(StoreContext);

  return useObserver(() => (
    <ErrorBoundary>
      <Parameter
        value={store.strike}
        step={0.25}
        setValue={store.setStrike}
        label={"Strike"}
      />
      <Parameter
        value={store.underlying}
        step={0.25}
        setValue={store.setUnderlying}
        label={"Underlying Price"}
      />
      <Parameter
        value={store.volatility}
        step={0.0001}
        setValue={store.setVolatility}
        label={"Volatility"}
      />
      <Parameter
        value={store.rate}
        step={0.0001}
        setValue={store.setRate}
        label={"Rate"}
      />
      <Parameter
        value={store.time}
        step={0.0001}
        setValue={store.setTime}
        label={"Time to Expiry"}
      />
      <Switch defaultChecked={false} onChange={store.setUseSolver} checkedChildren="Imply" unCheckedChildren={"Price"} />
      <br />
      <Parameter
        value={store.solverPrice}
        step={0.0001}
        setValue={(x) => {
          if (store.useSolver) {
            store.setSolverPrice(x)
          } else {
            throw new Error("Toggle PRICE -> IMPLY to use Implied Volatility Solver")
          }
        }}
        label={"Solver Price"}
      />
    </ErrorBoundary>
  ));
};

const Results = () => {
  const store = React.useContext(StoreContext);
  return useObserver(() => {
    const bs = store.bs;
    const columns = [
      {
        title: "",
        dataIndex: "kind",
        key: "kind",
        render: (x) => x.toUpperCase(),
      },
      {
        title: "C",
        dataIndex: "call",
        key: "call",
        render: (x) => Math.round(x * 1000000) / 1000000,
      },
      {
        title: "P",
        dataIndex: "put",
        key: "put",
        render: (x) => Math.round(x * 1000000) / 1000000,
      },
    ];
    return (
      <Table
        columns={columns}
        dataSource={bs}
        size="small"
        pagination={false}
      />
    );
  });
};

const BSPricer = () => {
  return (
    <StoreProvider>
      <Row>
        <Col span={12}>
          <Parameters />
        </Col>
        <Col span={12}>
          <Results />
        </Col>
      </Row>
    </StoreProvider>
  );
};

export default BSPricer;
