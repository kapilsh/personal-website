import React, { useState } from "react";
import { PageHeader, Row, Col, InputNumber, Form, Statistic } from "antd";

import { BS } from "../../pricingmodels/BlackScholes";

const layout = {
  labelCol: {
    span: 16,
  },
  wrapperCol: {
    span: 8,
  },
};

const ParametersForm = (props) => {
  console.log(props);

  const onFinish = (values) => {
    console.log("Success:", values);
  };

  const onFinishFailed = (errorInfo) => {
    console.log("Failed:", errorInfo);
  };

  const strike = props.strike;
  const underlyingPrice = props.underlying;
  const volatility = props.volatility;
  const riskFreeRate = props.riskFreeRate;
  const t = props.time;

  return (
    <Form
      {...layout}
      name="basic"
      initialValues={{
        remember: true,
      }}
      onFinish={onFinish}
      onFinishFailed={onFinishFailed}
    >
      <Form.Item
        label="Strike"
        name="strike"
        rules={[
          {
            required: true,
            message: "Please input strike price!",
          },
        ]}
      >
        <InputNumber
          min={0}
          max={10000}
          step={0.25}
          value={strike}
          onChange={(value) => {
            console.log(`Value changed to ${value}`);
          }}
        />
      </Form.Item>
      <Form.Item
        label="Underlying"
        name="underlying"
        rules={[
          {
            required: true,
            message: "Please input underlying price!",
          },
        ]}
      >
        <InputNumber
          min={0}
          max={10000}
          step={0.25}
          value={underlyingPrice}
          onChange={(value) => {
            console.log(`Value changed to ${value}`);
          }}
        />
      </Form.Item>
      <Form.Item
        label="Volatility"
        name="volatility"
        rules={[
          {
            required: true,
            message: "Please input volatility!",
          },
        ]}
      >
        <InputNumber
          min={0}
          max={10.0}
          step={0.0001}
          value={volatility}
          onChange={(value) => {
            console.log(`Value changed to ${value}`);
          }}
        />
      </Form.Item>
      <Form.Item
        label="Time To Expiration (Year)"
        name="yte"
        rules={[
          {
            required: true,
            message: "Please input time to expiration!",
          },
        ]}
      >
        <InputNumber
          min={0}
          max={10.0}
          step={0.0001}
          value={t}
          onChange={(value) => {
            console.log(`Value changed to ${value}`);
          }}
        />
      </Form.Item>
      <Form.Item
        label="Risk Free Rate"
        name="rate"
        rules={[
          {
            required: true,
            message: "Please input risk free rate!",
          },
        ]}
      >
        <InputNumber
          min={0}
          max={10.0}
          step={0.0001}
          value={riskFreeRate}
          onChange={(value) => {
            console.log(`Value changed to ${value}`);
          }}
        />
      </Form.Item>
    </Form>
  );
};

const Parameters = (props) => {
  const [strike, setStrike] = useState(props.strike);
  const underlyingPrice = props.underlying;
  const volatility = props.volatility;
  const riskFreeRate = props.riskFreeRate;
  const t = props.time;
  const recalculate = props.recalculate;
  return (
    <Row>
      <Col span={16}>
        <Form.Item label="Strike" name="strike" />
      </Col>
      <Col span={8}>
        <InputNumber
          min={0}
          max={10000}
          step={0.25}
          value={strike}
          onChange={(value) => {
            console.log(`Value changed to ${value}`);
            setStrike(value);
            recalculate(props);
          }}
        />
      </Col>
    </Row>
  );
};

class Options extends React.Component {
  render() {
    return (
      <>
        <PageHeader
          className="site-page-header"
          title="Options Playground"
          subTitle="Playground to play with options pricing models"
        />
        <Row>
          <Col span={12}>
            {/* <ParametersForm strike={100.0} underlying={100.0} /> */}
            <Parameters strike={100.0} recalculate={(p) => console.log(p)} />
          </Col>
          <Col></Col>
        </Row>
      </>
    );
  }
}

export default Options;
