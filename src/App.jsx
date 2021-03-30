import React from "react";
import { HashRouter as Router } from "react-router-dom";
import Layout from "./components/ApplicationLayout";
import ReactGA from "react-ga";
import "./App.css";

import { createBrowserHistory } from "history";

const trackingId = "UA-92270634-1";
ReactGA.initialize(trackingId);

const history = createBrowserHistory();


class App extends React.Component {
  render() {
    return (
      <Router onUpdate={() => window.scrollTo(0, 0)} history={history}>
        <Layout />
      </Router>
    );
  }
}

export default App;
