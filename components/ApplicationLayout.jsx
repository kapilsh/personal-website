import React from "react";

import { Layout, Menu, Space, PageHeader, Tag, Divider, Card } from "antd";

import Sidebar from "./Sidebar";
import Content from "./Content";
import posts from "../posts";
import { Link } from "react-router-dom";
import { TagOutlined } from "@ant-design/icons";

const { Sider } = Layout;
const { Meta } = Card;

const PostTag = ({ text }) => (
  <Tag color={"#1890ff"}>
    <Space />
    <span>#{text.replaceAll(" ", "").toLowerCase()}</span>
    <Space />
  </Tag>
);

class AppLayout extends React.Component {
  constructor(props) {
    super(props);
    this.state = { collapsed: false };
  }

  render() {
    return (
      <Layout style={{ minHeight: "100vh" }}>
        <Sidebar />
        <Content collapsed={this.state.collapsed} />
        <Sider
          width={300}
          breakpoint="lg"
          collapsedWidth={0}
          onCollapse={(collapsed, type) => {
            this.setState({ collapsed: collapsed });
          }}
          style={{
            overflow: "auto",
            height: "100vh",
            position: "fixed",
            right: 0,
          }}
        >
          <PageHeader className="site-page-header" title="More Posts" />
          {posts.map((post, index) => {
            const link = post.title.replaceAll(" ", "-").toLowerCase();
            return (
              <Link to={`/posts/${link}`}>
                <Card
                  hoverable
                  size={"small"}
                  style={{ width: "100%" }}
                  key={index}
                >
                  <p>{post.title}</p>
                  <p>
                    {post.tags.map((t, i) => {
                      return <PostTag text={t} key={i} />;
                    })}
                  </p>
                </Card>
              </Link>
            );
          })}
        </Sider>
      </Layout>
    );
  }
}

export default AppLayout;
