import React from "react";
import { Layout } from "antd";
const { Content, Footer } = Layout;
import { Route, Switch, Redirect } from "react-router-dom";

import Home from "./home/Home";
import About from "./about/About";
import Blog from "./blog/Blog";
import NotFound from "./NotFound";
import PricingPanel from "./options/PricingPanel";

class ContentPage extends React.Component {
  render() {
    const marginRight = this.props.collapsed ? 0 : 300;
    return (
      <Layout style={{ marginLeft: 80, marginRight: marginRight }}>
        <Content style={{ margin: "10px", padding: "10px" }}>
          <Switch>
            <Route exact path={"/"} component={Home} />
            <Route path={"/about"} component={About} />
            <Route path={"/options"} component={PricingPanel} />
            <Route path={"/posts/:postId"} component={Blog} />
            <Route path="/404" component={NotFound} />
            <Redirect path="*" to="/404" />
          </Switch>
        </Content>
        <Footer style={{ textAlign: "center" }}>
          Kapil Sharma © {new Date().getFullYear()} | Data Scientist, Math Geek,
          Programmer, Musician
        </Footer>
      </Layout>
    );
  }
}

export default ContentPage;
