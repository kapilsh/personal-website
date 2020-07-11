import React from "react";
import { Typography, Alert } from "antd";
import { CodeBlock, dracula } from "react-code-blocks";
import ZmqProxyImage from "../../../static/zmq_proxy.png";
import ForwarderImage from "../../../static/forwarder.png";
import { GithubFilled } from "@ant-design/icons";

const { Title, Paragraph } = Typography;

const publisherJava = `package zmq_forwarder;

import java.util.Random;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class Publisher {
    public static void main(String[] args) throws Exception {
		Context context = ZMQ.context(1);
		Socket publisher = context.socket(ZMQ.PUB);

		Random rand = new Random(System.currentTimeMillis());
		int serverNo = rand.nextInt(10000);

		publisher.connect("tcp://127.0.0.1:9999");

		System.out.println(String.format("Server : %s", serverNo));

		try {
			while (!Thread.currentThread().isInterrupted()) {
				String topic = String.format("%s", rand.nextInt(10));
				String payload = String.format("Server#%s", serverNo);
				publisher.sendMore(topic);
				publisher.send(payload);
				System.out.println("Sending: " + payload + " on Channel " + topic);
				Thread.sleep(250);
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
		} finally {
			publisher.close();
			context.term();
		}
	}
}`;

const subscriberJava = `package zmq_forwarder;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class Subscriber {
	public static void main(String[] args) {
		Context context = ZMQ.context(1);
		Socket subscriber = context.socket(ZMQ.SUB);

		subscriber.connect("tcp://127.0.0.1:8888");
		subscriber.subscribe("1");

		try {
			while (!Thread.currentThread().isInterrupted()) {
				String topic = subscriber.recvStr();
				String payload = subscriber.recvStr();
				System.out.println(topic + " : " + payload);
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
		} finally {
			subscriber.close();
			context.term();
		}

	}
}`;

const proxyJava = `package zmq_forwarder;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class Forwarder {
	public static void main(String[] args) {

		Context context = ZMQ.context(1);
		Socket frontend = context.socket(ZMQ.SUB);
		frontend.bind("tcp://*:9999");
		frontend.subscribe("".getBytes());

		Socket backend = context.socket(ZMQ.PUB);
		backend.bind("tcp://*:8888");

		try {
			System.out.println("Starting forwarder");
			ZMQ.proxy(frontend, backend, null);
		} catch (Exception e) {
			System.err.println(e.getMessage());
		} finally {
			frontend.close();
			backend.close();
			context.term();
		}

	}

}`;

const JavaSnippet = (props) => {
  return (
    <div
      style={{
        fontFamily: "Source Code Pro",
      }}
    >
      <CodeBlock
        text={props.snippet}
        language={"java"}
        showLineNumbers={!props.hideLines}
        theme={dracula}
        wrapLines
      />
      <br />
    </div>
  );
};

class ZeroMQProxy extends React.Component {
  render() {
    return (
      <>
        <Typography>
          <Paragraph>
            <a href="http://zeromq.org/">ZeroMQ</a> is my favorite message
            passing and networking library. It has bindings for almost all major
            languages and it's super convenient to build polyglot distributed
            network applications with it. Also, ZeroMQ documentation and
            examples are very exhaustive.
          </Paragraph>
          <Title level={3}>Dynamic Discovery Problem</Title>
          <Paragraph>
            I was recently struck with a problem where I had multiple ZeroMQ
            consumers connecting to multiple ZeroMQ producers. I needed a way to
            have a static port for consuming messages since adding new consumers
            and producers was getting hard to maintain. This is because all
            consumers needed to change whenever a new producer was added to the
            system. That's where I discovered the{" "}
            <a href="http://zguide.zeromq.org/java:chapter2#The-Dynamic-Discovery-Problem">
              Dynamic Discovery Problem
            </a>
            , which is precisely the problem that I described above. ZeroMQ
            documentation suggests using pub-sub proxy as a simple solution to
            this problem. In the post, I will create a working example for
            pub-sub proxy in Java.
          </Paragraph>
          <img
            alt="zmq proxy"
            src={ZmqProxyImage}
            style={{
              width: "45%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
              backgroundColor: "white",
            }}
          />
          <br />
          <Title level={3}>Publisher</Title>
          <Paragraph>
            Let’s create a Publisher that randomly sends payloads with topics
            between 0-9. The publisher will also get a randomly generated server
            number. Then, we can start multiple of these publishers and connect
            them to the Proxy or Forwarder socket.
          </Paragraph>
        </Typography>
        <JavaSnippet
          snippet={`Context context = ZMQ.context(1);
Socket publisher = context.socket(ZMQ.PUB);

Random rand = new Random(System.currentTimeMillis());
int serverNo = rand.nextInt(10000);

publisher.connect("tcp://127.0.0.1:9999"); // Connect to Proxy Server`}
        />
        <Alert
          message="NOTE"
          description="The SUBSCRIBER endpoint of the Proxy is running on port 9999"
          type="info"
          showIcon
        />
        <br />
        <Typography>
          <Paragraph>Here is the complete code for Publisher</Paragraph>
        </Typography>
        <JavaSnippet snippet={publisherJava} />
        <Typography>
          <Title level={3}>Subscriber</Title>
          <Paragraph>
            Now, we can create a Subcriber socket that subscribes to a specific
            topic (`1` in this case). We connect the subscriber to the Proxy
            server on a static port.
          </Paragraph>
          <Alert
            message="NOTE"
            description="The PUBLISHER endpoint of the Proxy is running on port 8888"
            type="info"
            showIcon
          />
          <br />
        </Typography>
        <JavaSnippet
          snippet={`Context context = ZMQ.context(1);
Socket subscriber = context.socket(ZMQ.SUB);
subscriber.connect("tcp://127.0.0.1:8888");
subscriber.subscribe("1");`}
        />
        <Typography>
          <Paragraph>
            Now we have all producers and consumers connecting to their
            respective static ports and Proxy will automatically route the
            producers to consumers. ZeroMQ also supports Subscription
            Forwarding. Hence, whatever topic consumer subscribes to
            automatically gets forwarded to producer sockets.
          </Paragraph>
          <Alert
            message="NOTE"
            description="This subscriber socket will receive all messages with topic 1 from all publishers."
            type="info"
            showIcon
          />
          <br />
          <Paragraph>Here is the complete code for Subscriber</Paragraph>
        </Typography>
        <JavaSnippet snippet={subscriberJava} />
        <Typography>
          <Title level={3}>Proxy</Title>
          <Paragraph>
            Finally, here we have the Pub-Sub Proxy. As we can see, proxy binds
            to all producers on port 9999 and all consumers on port 8888.
          </Paragraph>
        </Typography>
        <JavaSnippet
          snippet={`Socket frontend = context.socket(ZMQ.SUB);
frontend.bind("tcp://*:9999");
frontend.subscribe("".getBytes());

Socket backend = context.socket(ZMQ.PUB);
backend.bind("tcp://*:8888");

ZMQ.proxy(frontend, backend, null); // Create Proxy or Forwarder`}
        />
        <Alert
          message="NOTE"
          description="The `null` third parameter can be another socket where you can sniff the traffic; however, that is a topic for another day."
          type="info"
          showIcon
        />
        <br />
        <JavaSnippet snippet={proxyJava} />
        <Typography>
          <Title level={3}>Working Project</Title>
          <Paragraph>
            You can find a maven based project for this post on my{" "}
            <a
              href={
                "https://github.com/kapilsh/mini-projects/tree/master/java/zmq_forwarder"
              }
            >
              <GithubFilled /> Github
            </a>
          </Paragraph>
          <Paragraph>
            Build the project by running "mvn clean package" in the terminal.
            Post build, you should see three jar files in the target folder:
            <ul>
              <li>publisher.jar</li>
              <li>subscriber.jar</li>
              <li>proxy.jar</li>
            </ul>
          </Paragraph>
        </Typography>
        <div
          style={{
            fontFamily: "Source Code Pro",
          }}
        >
          <CodeBlock
            text={`ls target | grep jar\nproxy.jar\nublisher.jar\nsubscriber.jar\nzmq_forwarder-0.0.1.jar`}
            language={"bash"}
            showLineNumbers={false}
            theme={dracula}
            wrapLines
          />
        </div>
        <br />
        <Typography>
          <Title level={3}>My Awesome Tmux Window</Title>
          <Paragraph>Run all the jar files as below:</Paragraph>
          <img
            alt="my awesome tmux window"
            src={ForwarderImage}
            style={{
              width: "100%",
            }}
          />
          <br />
          <br />
          <Alert
            message="NOTE"
            description="Consumer is receiving all messages from both producers for topic 1"
            type="info"
            showIcon
          />
          <br />
          <Title level={3}>Conclusion</Title>
          <Paragraph>
            ZMQ Proxy provides a neat way to connect multiple ZeroMQ subscribers
            to multiple ZeroMQ publishers and solves the Dynamic Discovery
            Problem.
          </Paragraph>
        </Typography>
      </>
    );
  }
}

export default ZeroMQProxy;
