import React from "react";
import { Card, Row, Col, Typography } from "antd";
import profileImage from "../../../static/profile.png";

const { Meta } = Card;
const { Paragraph } = Typography;

class About extends React.Component {
  render() {
    return (
      <Row>
        <Col span={24}>
          <Card
            hoverable
            style={{ width: "100%" }}
            cover={<img alt="profile picture" src={profileImage} />}
          >
            <Meta
              title="Kapil Sharma | Data Scientist, Math Geek, Programmer, Musician"
              description={
                <Typography>
                  <Paragraph>
                    Hi, I’m Kapil Sharma. I am a Chicago-based Quantitative
                    Researcher and Data Scientist working in the High Frequency
                    Trading industry. Before that, I studied graduate-level
                    Mathematics and Quantitative Finance at Columbia University
                    in New York City. My professional interests include High
                    Frequency Trading, Quantitative Finance, Mathematics,
                    Algorithms, Machine Learning, and Programming.
                  </Paragraph>
                  <Paragraph>
                    When I am not coding and white-boarding equations, I love to
                    play music and watch football (aka soccer). I am a big
                    Chelsea F.C. fan. #KTBFFH I love to connect with like-minded
                    people. So, give me a shout!
                  </Paragraph>
                </Typography>
              }
            />
            <hr />
            <Typography>
              <Paragraph>On this blog, I post:</Paragraph>
              <ul>
                <li>
                  Collection of articles and notes on interesting Math problems
                  that I encounter on a daily basis
                </li>
                <li>
                  My experiments with Machine Learning, Data Science,
                  Algorithmic Trading, Trading System Design, Time-series
                  Analysis, etc.
                </li>
                <li>
                  Snippets and discussions related to new (for me) programming
                  techniques, mostly in C++, Python, and Java
                </li>
              </ul>
            </Typography>
          </Card>
        </Col>
      </Row>
    );
  }
}

export default About;
