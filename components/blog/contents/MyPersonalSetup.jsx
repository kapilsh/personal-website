import React from "react";
import { Typography } from "antd";
import { CodeBlock, dracula } from "react-code-blocks";

const { Title, Paragraph } = Typography;

const CodeSnippet = (props) => {
  return (
    <>
      <Typography>
        <br />
        <Title level={3}>{props.title}</Title>
      </Typography>
      <div
        style={{
          fontFamily: "Source Code Pro",
        }}
      >
        <CodeBlock
          text={props.text}
          language={"bash"}
          showLineNumbers={true}
          theme={dracula}
          wrapLines
        />
      </div>
    </>
  );
};

const zshSnippet = `$ sudo yum install -y zsh
$ sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

$ vim ~/.zshrc

....

ZSH_THEME="candy"
....`;

const gitSnippet = `$ sudo yum install zlib-devel -y
$ mkdir downloads && cd downloads
$ wget https://github.com/git/git/archive/v2.20.1.tar.gz
$ tar -xzvf v2.20.1.tar.gz
$ cd git-2.20.1 
$ make configure
$ ./configure --prefix=\${HOME}/local
$ make all && make install
$ vim ~/.zshrc

# Add this at the end of the file
export PATH=\${HOME}/local/bin:\${PATH}`;

const tmuxSnippet = `# Download

$ cd ~/downloads

$ wget https://github.com/tmux/tmux/releases/download/2.8/tmux-2.8.tar.gz
$ wget https://github.com/libevent/libevent/releases/download/release-2.1.8-stable/libevent-2.1.8-stable.tar.gz
$ wget https://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.1.tar.gz

# Extract

$ tar -xzvf ncurses-6.1.tar.gz
$ tar -zxvf libevent-2.1.8-stable.tar.gz
$ tar -xzvf tmux-2.8.tar.gz

# Install libevents
$ cd ~/downloads/libevent-2.1.8-stable
$ ./configure --prefix=$HOME/local --disable-shared
$ make -j4
$ make install

# Install ncurses

$ cd ~/downloads/ncurses-6.1
$ ./configure --prefix=$HOME/local
$ make -j4
$ make install

# Install tmux

$ cd ~/downloads/tmux-2.8
$ ./configure --prefix=\${HOME}/local CFLAGS="-I$HOME/local/include -I$HOME/local/include/ncurses" LDFLAGS="-L$HOME/local/lib -L$HOME/local/include/ncurses -L$HOME/local/include"
$ make -j6 && make install
`;

const dockerSnippet = `$ sudo yum install -y hdf5-devel java-1.8.0-openjdk 
$ sudo yum remove docker \\
                  docker-client \\
                  docker-client-latest \\
                  docker-common \\
                  docker-latest \\
                  docker-latest-logrotate \\
                  docker-logrotate \\
                  docker-selinux \\
                  docker-engine-selinux \\
                  docker-engine
$ sudo yum install -y yum-utils \\
  device-mapper-persistent-data \\
  lvm2
$ sudo yum-config-manager \
    --add-repo \\
    https://download.docker.com/linux/centos/docker-ce.repo
$ sudo yum install docker-ce -y
$ sudo systemctl start docker
$ sudo usermod -aG docker $USER
$ sudo reboot
`;

const pythonSnippet = `$ cd ~/downloads && wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x Anaconda3-2018.12-Linux-x86_64.sh 
$ ./Anaconda3-2018.12-Linux-x86_64.sh
$ vim ~/.zshrc
# Add this at the end of the file
$ export PATH=\${HOME}/anaconda3/bin:\${PATH}
`;

class MyPersonalSetup extends React.Component {
  render() {
    return (
      <>
        <Typography>
          <Paragraph>
            This is more of a personal to-do list to setup a new linux
            development environment. Hope, this helps others to setup their own.
          </Paragraph>
        </Typography>
        <CodeSnippet title={"ZSH"} text={zshSnippet} />
        <CodeSnippet title={"GIT"} text={gitSnippet} />
        <CodeSnippet title={"TMUX"} text={tmuxSnippet} />
        <CodeSnippet title={"DOCKER"} text={dockerSnippet} />
        <CodeSnippet title={"PYTHON"} text={pythonSnippet} />
      </>
    );
  }
}

export default MyPersonalSetup;
