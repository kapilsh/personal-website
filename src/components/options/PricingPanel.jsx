import React from "react";
import { PageHeader, Row, Col, InputNumber, Form, Table, Tabs } from "antd";
import BSPricer from "./BSPricer";

const { TabPane } = Tabs;

const PricingTabs = () => (
  <Tabs defaultActiveKey="1">
    <TabPane tab="BS Pricer" key="1">
      <BSPricer />
    </TabPane>
  </Tabs>
);

const PricingPanel = () => {
  return (
    <>
      <PageHeader
        className="site-page-header"
        title="Options Playground"
        subTitle="Playground to play with options pricing models"
      />
      <PricingTabs />
    </>
  );
};

export default PricingPanel;
