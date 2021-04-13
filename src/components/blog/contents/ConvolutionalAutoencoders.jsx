import React from "react";
import {Collapse} from 'antd';
import {Typography} from "antd";
import {Alert} from 'antd';
import {LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label} from 'recharts';
import {PythonSnippet} from "../snippets/PythonSnippet";
import {BashSnippet} from "../snippets/BashSnippet";
import 'react-github-cards/dist/default.css';
import Image1 from "../../../../static/conv-autoencoder-1.png";
import Image2 from "../../../../static/conv-autoencoder-2.png";
import Image3 from "../../../../static/conv-autoencoder-3.png";
import Image4 from "../../../../static/conv-autoencoder-4.png";
import mlProjectsSvg from "../../../../static/ml-projects.svg"

const {Title, Paragraph} = Typography;
const {Panel} = Collapse;


const trainingLossJSON = [{'Epoch': 0, 'TrainingLoss': 0.4373},
    {'Epoch': 1, 'TrainingLoss': 0.2375},
    {'Epoch': 2, 'TrainingLoss': 0.2183},
    {'Epoch': 3, 'TrainingLoss': 0.208},
    {'Epoch': 4, 'TrainingLoss': 0.2024},
    {'Epoch': 5, 'TrainingLoss': 0.199},
    {'Epoch': 6, 'TrainingLoss': 0.1965},
    {'Epoch': 7, 'TrainingLoss': 0.1948},
    {'Epoch': 8, 'TrainingLoss': 0.1939},
    {'Epoch': 9, 'TrainingLoss': 0.1932},
    {'Epoch': 10, 'TrainingLoss': 0.1926},
    {'Epoch': 11, 'TrainingLoss': 0.1921},
    {'Epoch': 12, 'TrainingLoss': 0.1916},
    {'Epoch': 13, 'TrainingLoss': 0.1911},
    {'Epoch': 14, 'TrainingLoss': 0.1907},
    {'Epoch': 15, 'TrainingLoss': 0.1903},
    {'Epoch': 16, 'TrainingLoss': 0.1899},
    {'Epoch': 17, 'TrainingLoss': 0.1895},
    {'Epoch': 18, 'TrainingLoss': 0.1892},
    {'Epoch': 19, 'TrainingLoss': 0.189},
    {'Epoch': 20, 'TrainingLoss': 0.1888},
    {'Epoch': 21, 'TrainingLoss': 0.1885},
    {'Epoch': 22, 'TrainingLoss': 0.1882},
    {'Epoch': 23, 'TrainingLoss': 0.188},
    {'Epoch': 24, 'TrainingLoss': 0.1878},
    {'Epoch': 25, 'TrainingLoss': 0.1875},
    {'Epoch': 26, 'TrainingLoss': 0.1873},
    {'Epoch': 27, 'TrainingLoss': 0.187},
    {'Epoch': 28, 'TrainingLoss': 0.1867},
    {'Epoch': 29, 'TrainingLoss': 0.1864},
    {'Epoch': 30, 'TrainingLoss': 0.1861},
    {'Epoch': 31, 'TrainingLoss': 0.1857},
    {'Epoch': 32, 'TrainingLoss': 0.1854},
    {'Epoch': 33, 'TrainingLoss': 0.185},
    {'Epoch': 34, 'TrainingLoss': 0.1845},
    {'Epoch': 35, 'TrainingLoss': 0.184},
    {'Epoch': 36, 'TrainingLoss': 0.1836},
    {'Epoch': 37, 'TrainingLoss': 0.1834},
    {'Epoch': 38, 'TrainingLoss': 0.183},
    {'Epoch': 39, 'TrainingLoss': 0.1828},
    {'Epoch': 40, 'TrainingLoss': 0.1825},
    {'Epoch': 41, 'TrainingLoss': 0.1823},
    {'Epoch': 42, 'TrainingLoss': 0.1821},
    {'Epoch': 43, 'TrainingLoss': 0.1819},
    {'Epoch': 44, 'TrainingLoss': 0.1818},
    {'Epoch': 45, 'TrainingLoss': 0.1816},
    {'Epoch': 46, 'TrainingLoss': 0.1815},
    {'Epoch': 47, 'TrainingLoss': 0.1813},
    {'Epoch': 48, 'TrainingLoss': 0.1812},
    {'Epoch': 49, 'TrainingLoss': 0.1811}];

class ConvolutionalAutoencoders extends React.Component {
    render() {
        return (<>
            <Typography>
                <Paragraph>
                    The success of Convolutional Neural Networks at image classification is well known but the same
                    conceptual backbone can be used for other image tasks as well, for example image
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
                    After defining the datasets, we can define data loaders, which will feed batches of images into our neural
                    network:
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
                            Encoder: Looks similar to the regular convolutional pyramid of CNN's
                        </li>
                        <li>
                            Decoder: Converts the narrow representation to wide, reconstructed image. It applies
                            multiple transpose convolutional layers to go from compressed representation to a regular
                            image
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
            <Collapse accordian>
                <Panel header="Training Logs" key="1">
                    <BashSnippet
                        text={"2020-09-07 20:30:13.750 | WARNING  | autoencoders.convolutional_autoencoder:__init__:91 - CUDA not available/enabled. Using CPU\n" +
                        "2020-09-07 20:30:13.751 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 1: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.26it/s]\n" +
                        "2020-09-07 20:30:23.810 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 1: Training loss 0.437312\n" +
                        "2020-09-07 20:30:23.810 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 2: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:09<00:00, 301.93it/s]\n" +
                        "2020-09-07 20:30:33.747 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 2: Training loss 0.237542\n" +
                        "2020-09-07 20:30:33.747 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 3: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:09<00:00, 302.29it/s]\n" +
                        "2020-09-07 20:30:43.671 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 3: Training loss 0.218295\n" +
                        "2020-09-07 20:30:43.672 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 4: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 294.49it/s]\n" +
                        "2020-09-07 20:30:53.859 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 4: Training loss 0.207977\n" +
                        "2020-09-07 20:30:53.859 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 5: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:09<00:00, 300.07it/s]\n" +
                        "2020-09-07 20:31:03.857 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 5: Training loss 0.202394\n" +
                        "2020-09-07 20:31:03.857 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 6: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.93it/s]\n" +
                        "2020-09-07 20:31:13.893 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 6: Training loss 0.199011\n" +
                        "2020-09-07 20:31:13.893 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 7: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 284.79it/s]\n" +
                        "2020-09-07 20:31:24.428 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 7: Training loss 0.196497\n" +
                        "2020-09-07 20:31:24.428 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 8: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 284.40it/s]\n" +
                        "2020-09-07 20:31:34.977 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 8: Training loss 0.194811\n" +
                        "2020-09-07 20:31:34.977 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 9: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 299.79it/s]\n" +
                        "2020-09-07 20:31:44.984 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 9: Training loss 0.193926\n" +
                        "2020-09-07 20:31:44.984 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 10: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 294.04it/s]\n" +
                        "2020-09-07 20:31:55.187 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 10: Training loss 0.193171\n" +
                        "2020-09-07 20:31:55.187 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 11: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.18it/s]\n" +
                        "2020-09-07 20:32:05.248 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 11: Training loss 0.192589\n" +
                        "2020-09-07 20:32:05.249 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 12: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.21it/s]\n" +
                        "2020-09-07 20:32:15.309 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 12: Training loss 0.192103\n" +
                        "2020-09-07 20:32:15.309 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 13: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 294.76it/s]\n" +
                        "2020-09-07 20:32:25.487 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 13: Training loss 0.191604\n" +
                        "2020-09-07 20:32:25.487 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 14: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 296.91it/s]\n" +
                        "2020-09-07 20:32:35.592 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 14: Training loss 0.191146\n" +
                        "2020-09-07 20:32:35.592 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 15: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.23it/s]\n" +
                        "2020-09-07 20:32:45.651 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 15: Training loss 0.190652\n" +
                        "2020-09-07 20:32:45.652 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 16: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.84it/s]\n" +
                        "2020-09-07 20:32:55.691 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 16: Training loss 0.190262\n" +
                        "2020-09-07 20:32:55.691 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 17: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.83it/s]\n" +
                        "2020-09-07 20:33:05.764 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 17: Training loss 0.189871\n" +
                        "2020-09-07 20:33:05.764 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 18: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.28it/s]\n" +
                        "2020-09-07 20:33:15.856 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 18: Training loss 0.189549\n" +
                        "2020-09-07 20:33:15.856 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 19: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 295.27it/s]\n" +
                        "2020-09-07 20:33:26.017 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 19: Training loss 0.189221\n" +
                        "2020-09-07 20:33:26.017 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 20: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 294.82it/s]\n" +
                        "2020-09-07 20:33:36.193 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 20: Training loss 0.189024\n" +
                        "2020-09-07 20:33:36.193 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 21: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.50it/s]\n" +
                        "2020-09-07 20:33:46.277 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 21: Training loss 0.188787\n" +
                        "2020-09-07 20:33:46.277 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 22: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.19it/s]\n" +
                        "2020-09-07 20:33:56.338 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 22: Training loss 0.188466\n" +
                        "2020-09-07 20:33:56.338 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 23: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.14it/s]\n" +
                        "2020-09-07 20:34:06.401 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 23: Training loss 0.18817\n" +
                        "2020-09-07 20:34:06.401 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 24: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.30it/s]\n" +
                        "2020-09-07 20:34:16.492 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 24: Training loss 0.188001\n" +
                        "2020-09-07 20:34:16.492 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 25: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 295.21it/s]\n" +
                        "2020-09-07 20:34:26.655 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 25: Training loss 0.187785\n" +
                        "2020-09-07 20:34:26.655 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 26: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.07it/s]\n" +
                        "2020-09-07 20:34:36.720 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 26: Training loss 0.18753\n" +
                        "2020-09-07 20:34:36.720 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 27: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 296.21it/s]\n" +
                        "2020-09-07 20:34:46.848 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 27: Training loss 0.187273\n" +
                        "2020-09-07 20:34:46.849 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 28: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.57it/s]\n" +
                        "2020-09-07 20:34:56.930 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 28: Training loss 0.187002\n" +
                        "2020-09-07 20:34:56.931 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 29: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.41it/s]\n" +
                        "2020-09-07 20:35:07.018 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 29: Training loss 0.186654\n" +
                        "2020-09-07 20:35:07.018 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 30: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.83it/s]\n" +
                        "2020-09-07 20:35:17.091 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 30: Training loss 0.186408\n" +
                        "2020-09-07 20:35:17.091 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 31: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.39it/s]\n" +
                        "2020-09-07 20:35:27.179 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 31: Training loss 0.18608\n" +
                        "2020-09-07 20:35:27.180 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 32: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 295.95it/s]\n" +
                        "2020-09-07 20:35:37.317 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 32: Training loss 0.185741\n" +
                        "2020-09-07 20:35:37.317 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 33: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.16it/s]\n" +
                        "2020-09-07 20:35:47.413 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 33: Training loss 0.185363\n" +
                        "2020-09-07 20:35:47.413 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 34: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.01it/s]\n" +
                        "2020-09-07 20:35:57.514 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 34: Training loss 0.184985\n" +
                        "2020-09-07 20:35:57.514 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 35: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.53it/s]\n" +
                        "2020-09-07 20:36:07.597 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 35: Training loss 0.184487\n" +
                        "2020-09-07 20:36:07.597 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 36: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.52it/s]\n" +
                        "2020-09-07 20:36:17.681 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 36: Training loss 0.184044\n" +
                        "2020-09-07 20:36:17.681 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 37: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 296.39it/s]\n" +
                        "2020-09-07 20:36:27.803 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 37: Training loss 0.183633\n" +
                        "2020-09-07 20:36:27.803 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 38: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.19it/s]\n" +
                        "2020-09-07 20:36:37.898 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 38: Training loss 0.183359\n" +
                        "2020-09-07 20:36:37.898 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 39: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 293.79it/s]\n" +
                        "2020-09-07 20:36:48.110 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 39: Training loss 0.183028\n" +
                        "2020-09-07 20:36:48.110 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 40: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.24it/s]\n" +
                        "2020-09-07 20:36:58.203 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 40: Training loss 0.182765\n" +
                        "2020-09-07 20:36:58.203 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 41: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 296.77it/s]\n" +
                        "2020-09-07 20:37:08.313 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 41: Training loss 0.182514\n" +
                        "2020-09-07 20:37:08.313 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 42: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 294.76it/s]\n" +
                        "2020-09-07 20:37:18.491 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 42: Training loss 0.182298\n" +
                        "2020-09-07 20:37:18.491 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 43: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.70it/s]\n" +
                        "2020-09-07 20:37:28.569 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 43: Training loss 0.182133\n" +
                        "2020-09-07 20:37:28.569 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 44: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 294.96it/s]\n" +
                        "2020-09-07 20:37:38.740 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 44: Training loss 0.181893\n" +
                        "2020-09-07 20:37:38.740 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 45: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 291.89it/s]\n" +
                        "2020-09-07 20:37:49.018 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 45: Training loss 0.181797\n" +
                        "2020-09-07 20:37:49.019 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 46: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.31it/s]\n" +
                        "2020-09-07 20:37:59.109 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 46: Training loss 0.181623\n" +
                        "2020-09-07 20:37:59.110 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 47: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 296.73it/s]\n" +
                        "2020-09-07 20:38:09.220 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 47: Training loss 0.181462\n" +
                        "2020-09-07 20:38:09.220 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 48: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.82it/s]\n" +
                        "2020-09-07 20:38:19.260 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 48: Training loss 0.181292\n" +
                        "2020-09-07 20:38:19.260 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 49: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 298.10it/s]\n" +
                        "2020-09-07 20:38:29.324 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 49: Training loss 0.181203\n" +
                        "2020-09-07 20:38:29.324 | INFO     | autoencoders.convolutional_autoencoder:train:100 - [EPOCH 50: Starting training\n" +
                        "100%|██████████| 3000/3000 [00:10<00:00, 297.16it/s]\n" +
                        "2020-09-07 20:38:39.420 | INFO     | autoencoders.convolutional_autoencoder:train:113 - [EPOCH 50: Training loss 0.181089"}/>
                </Panel>
            </Collapse>
            <br/>
            <ResponsiveContainer width="70%" height={450}>
                <LineChart
                    width={"70%"}
                    height={450}
                    data={trainingLossJSON}
                    margin={{
                        top: 5,
                        right: 30,
                        left: 35,
                        bottom: 35
                    }}
                    style={{
                        backgroundColor: "white",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                >
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis dataKey="Epochs">
                        <Label value="Epochs" position="bottom" offset={10} style={{color: "white"}}/>
                    </XAxis>
                    <YAxis>
                        <Label angle={270} value="Loss Value" position="left" offset={10} style={{"color": "white"}}/>
                    </YAxis>
                    <Tooltip/>
                    <Legend verticalAlign="top" height={36}/>
                    <Line type="monotone" dataKey="TrainingLoss" stroke="#8884d8" activeDot={{r: 8}}/>
                </LineChart>
            </ResponsiveContainer>
            <br/>
            <Alert
                message="NOTE"
                description="We can see that the training loss function decreases rapidly initially and then reaches a stable value after ~25 epochs"
                type="info"
                showIcon
            />
            <br/>
            <Typography>
                <Title level={3}>
                    Testing
                </Title>
                <Paragraph>
                    Let's look at how the network does with re-constructing the images. We will pass our test images
                    through the model and compare the input and output images. The code to test the images looks fairly
                    simple. We enable eval mode on the model and pass the image batch tensors through the model. The
                    output will be the re-constructed image tensors.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"if use_gpu:\n" +
            "    model = model.cuda()\n" +
            "model.eval()\n" +
            "for data, _ in data_provider.test:\n" +
            "    result = model(data)\n" +
            "    yield data, result"}/>
            <Typography>
                <Paragraph>
                    Let's look at some results:
                </Paragraph>
            </Typography>
            <img
                alt="Result 1"
                src={Image3}
                style={{
                    width: "80%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }}
            />
            <br/>
            <img
                alt="Result 2"
                src={Image4}
                style={{
                    width: "80%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }}
            />
            <br/>
            <Alert
                message="NOTE"
                description="We do a lot better with images with simple pixel structures such as T-shirts, dresses, sneakers. We don't do that well with intricate pixel structures such as heals and patterned T-shirts"
                type="info"
                showIcon
            />
            <br/>
            <Typography>
                <Title level={3}>
                    Github Link
                </Title>
                <Paragraph>
                    You can access the full project on my Github:
                </Paragraph>
                <a href={"https://github.com/kapilsh/ml-projects"}>
                    <img
                        alt="ML Projects Repo"
                        src={mlProjectsSvg}
                        style={{
                            width: "35%",
                            display: "block",
                            marginLeft: "auto",
                            marginRight: "auto",
                        }}
                    />
                </a>
            </Typography>
        </>);
    }
}

export default ConvolutionalAutoencoders;