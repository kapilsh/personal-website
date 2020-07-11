import React from "react";
import { Result, Button } from "antd";
import { HomeFilled } from "@ant-design/icons";
import Image404 from "../static/404.jpg";
import { Link } from "react-router-dom";

class NotFound extends React.Component {
  render() {
    return (
      <>
        <Result
          status="404"
          title="404"
          subTitle="Sorry, the page you visited does not exist."
          extra={
            <Button type="primary">
              <Link to={"/"}>
                <HomeFilled />
              </Link>
            </Button>
          }
        />
        <img
          src={Image404}
          style={{
            width: "70%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        />
      </>
    );
  }
}

export default NotFound;
