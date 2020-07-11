import React from "react";
import { Router } from "react-router-dom";
import Layout from "./components/ApplicationLayout";
import { Provider } from "react-redux";
import ReactGA from "react-ga";
import store from "./store";
import "./App.css";

import { createBrowserHistory } from "history";

const trackingId = "UA-92270634-1";
ReactGA.initialize(trackingId);

const history = createBrowserHistory();

history.listen((location) => {
  ReactGA.set({ page: location.pathname });
  ReactGA.pageview(location.pathname);
});

class App extends React.Component {
  render() {
    return (
      <Provider store={store}>
        <Router
          onUpdate={() => window.scrollTo(0, 0)}
          history={history}
          basename={`${process.env.PUBLIC_URL}`}
        >
          <Layout />
        </Router>
      </Provider>
    );
  }
}

export default App;
