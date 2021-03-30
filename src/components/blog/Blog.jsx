import React, { useEffect } from "react";
import { Typography, Card, Tag, Space } from "antd";
import posts from "../../posts";
import HyvorTalk from "hyvor-talk-react";
import { TagOutlined, CalendarOutlined } from "@ant-design/icons";

import "./Blog.css";
import NotFound from "../NotFound";

const { Title } = Typography;

const PostTag = ({ text }) => (
  <Tag color={"#1890ff"}>
    <Space />
    <TagOutlined />
    <span>{text}</span>
    <Space />
  </Tag>
);

const BlogDate = ({ date }) => (
  <Space>
    <CalendarOutlined />
    {date}
  </Space>
);

const Blog = ({ match }) => {
  const {
    params: { postId },
  } = match;
  const post = posts.filter(
    (p) => p.title.split(" ").join("-").toLowerCase() === postId
  )[0];
  if (post == null) {
      return (
          <NotFound />
          )
  }
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);
  return (
    <Card
      style={{ width: "100%" }}
      cover={
        <img
          alt={post.image}
          src={post.image}
          style={{
            width: "100%",
            height: "350px",
            objectFit: "cover",
          }}
        />
      }
    >
      <Typography>
        <Title>{post.title}</Title>
        <Title level={2}>{post.description}</Title>
        <Title level={3}>
          {
            <Tag>
              <BlogDate date={post.date} />
            </Tag>
          }
          {post.tags.map((t, i) => {
            return <PostTag text={t} key={i} />;
          })}
          <hr />
        </Title>
      </Typography>
      {post.component}
      <hr />
      <HyvorTalk.Embed websiteId={1094} id={post.title} />
    </Card>
  );
};

export default Blog;
