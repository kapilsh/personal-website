import React from "react";
import {Collapse} from 'antd';
import {Typography} from "antd";
import {Alert} from 'antd';
import {LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label} from 'recharts';
import {Link} from "react-router-dom";
import {PythonSnippet} from "../snippets/PythonSnippet";
import {Code, dracula} from "react-code-blocks";
import {BashSnippet} from "../snippets/BashSnippet";
import {UserCard, RepoCard} from 'react-github-cards';
import 'react-github-cards/dist/default.css';
import Image1 from "../../../../static/conv-autoencoder-1.png";
import Image2 from "../../../../static/conv-autoencoder-2.png";

const {Title, Paragraph} = Typography;
const {Panel} = Collapse;

class ConvolutionalAutoencoders extends React.Component {
    render() {
        return (<>
            <Typography>
                <Paragraph>
                    The success of Convolutional Neural Networks at image classification is well known but the same
                    conceptual backbone can be use for other image tasks as well, for example image
                    compression. In this post, I will use Convolutional <a
                    href={"https://en.wikipedia.org/wiki/Autoencoder"}>Autoencoder</a> to re-generate compressed images
                    for the <a href={"https://github.com/zalandoresearch/fashion-mnist"}>Fashion MNIST</a> dataset.
                    Let's look at a few random images:
                </Paragraph>
                <img
                    alt="FMNIST Images"
                    src={Image1}
                    style={{
                        width: "80%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <Title level={3}>Setup</Title>
                <Paragraph>
                    The FMNIST dataset is available directly through the torchvision package. We can load it directly
                    into our environment using torchvision datasets module:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"from torchvision import transforms, datasets\n" +
            "\n" +
            "transform = transforms.ToTensor()\n" +
            "train_data = datasets.FashionMNIST(root=root_dir, train=True,\n" +
            "                                   download=True,\n" +
            "                                   transform=transform)\n" +
            "test_data = datasets.FashionMNIST(root=root_dir, train=False,\n" +
            "                                  download=True,\n" +
            "                                  transform=transform)\n"}/>
            <Typography>
                <Paragraph>
                    After defining the datasets, we can define data loaders, which will feed data into our neural
                    network in batches:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"from torch.utils.data import DataLoader\n" +
            "num_workers = 0\n" +
            "batch_size = 20\n" +
            "\n" +
            "train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)\n" +
            "test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)"}/>
            <Typography>
                <Title level={3}>Architecture</Title>
                <Paragraph>
                    The architecture consists of two segments:
                    <ol>
                        <li>
                            Encoder: Looks similar to the the regular convolutional pyramid of CNN's
                        </li>
                        <li>
                            Decoder: Converts the narrow representation to wide, reconstructed image. It applies
                            multiple transpose convolutional layers to go from compressed representation to a regular
                            image.
                        </li>
                    </ol>
                    Below I define the two set of layers using PyTorch nn.Module:
                </Paragraph>
            </Typography>
            <Typography>
                <Title level={4}>Encoder</Title>
                <Paragraph>
                    During the encoding phase, we pass the images through 2 convolutional layers, each followed by a max
                    pool layer. The final dimension of the encoded image is 4 channels of 7 x 7 matrices.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"class Encoder(nn.Module):\n" +
            "    def __init__(self):\n" +
            "        super().__init__()\n" +
            "        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=(1, 1))\n" +
            "        self.max_pool1 = nn.MaxPool2d(2, 2)\n" +
            "        self.conv2 = nn.Conv2d(16, 4, (3, 3), padding=(1, 1))\n" +
            "        self.max_pool2 = nn.MaxPool2d(2, 2)\n" +
            "\n" +
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n" +
            "        x = self.conv1(x)\n" +
            "        x = self.max_pool1(x)\n" +
            "        x = self.conv2(x)\n" +
            "        x = self.max_pool2(x)\n" +
            "        return x"}/>
            <Typography>
                <Title level={4}>Decoder</Title>
                <Paragraph>
                    During the decoding phase, we pass the decoded image through transpose convolutional layers to
                    increase the dimensions along width and height while bringing the number of channels down from 4 to
                    1.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"class Decoder(nn.Module):\n" +
            "    def __init__(self):\n" +
            "        super().__init__()\n" +
            "        self.t_conv1 = nn.ConvTranspose2d(4, 16, (2, 2), stride=(2, 2))\n" +
            "        self.t_conv2 = nn.ConvTranspose2d(16, 1, (2, 2), stride=(2, 2))\n" +
            "\n" +
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n" +
            "        x = self.t_conv1(x)\n" +
            "        x = functional.relu(x)\n" +
            "        x = self.t_conv2(x)\n" +
            "        x = torch.sigmoid(x)\n" +
            "        return x"}/>
            <Typography>
                <Title level={4}>Full Network</Title>
                <Paragraph>
                    In the full network, we combine both layers where encoder layer feeds into the decoder layer.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"class AutoEncoder(nn.Module):\n" +
            "    def __init__(self):\n" +
            "        super().__init__()\n" +
            "        self.encoder = Encoder()\n" +
            "        self.decoder = Decoder()\n" +
            "\n" +
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n" +
            "        x = self.encoder(x)\n" +
            "        x = self.decoder(x)\n" +
            "        return x"}/>
            <img
                alt="Architecture"
                src={Image2}
                style={{
                    width: "60%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }}
            />
            <Typography>
                <Title level={3}>Training</Title>
                <Paragraph>
                    During the training phase, we pass batches of images to our network. We finally compare the actual
                    re-constructed image with the original image using the MSE Loss function to check final pixel loss.
                    We optimize for minimum loss between original and re-constructed image using the Adam optimizer. We
                    train the model for a few epochs and stop after the loss function doesn't show signs of decreasing
                    further. Here's how the training loop looks like:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"model = AutoEncoder()\n" +
            "if use_gpu:\n" +
            "    model = model.cuda()\n" +
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n" +
            "criterion = nn.MSELoss()\n" +
            "train_losses = []\n" +
            "for epoch in range(1, n_epochs + 1):\n" +
            "    logger.info(f\"[EPOCH {epoch}: Starting training\")\n" +
            "    train_loss = 0.0\n" +
            "    batches = len(train_loader)\n" +
            "    for data, _ in tqdm(train_loader, total=batches):\n" +
            "        optimizer.zero_grad()\n" +
            "        if use_gpu:\n" +
            "            data = data.cuda() # transfer data to gpu\n" +
            "        output = model(data) # calculate predicted value\n" +
            "        loss = criterion(output, data) # calculate loss function\n" +
            "        loss.backward() # back propagation\n" +
            "        optimizer.step() # take an optimizer step\n" +
            "        train_loss += loss.item() * data.size(0)\n" +
            "    train_loss = train_loss / len(train_loader) # calculate average loss\n" +
            "    logger.info(\n" +
            "        f\"[EPOCH {epoch}: Training loss {np.round(train_loss, 6)}\")\n" +
            "    train_losses.append(train_loss)"}/>
        </>);
    }
}

export default ConvolutionalAutoencoders;