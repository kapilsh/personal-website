import React, {PureComponent} from "react";
import {Collapse} from 'antd';
import {Typography} from "antd";
import {Alert} from 'antd';
import {LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label} from 'recharts';
import {Link} from "react-router-dom";
import {PythonSnippet} from "../snippets/PythonSnippet";
import {Code, dracula} from "react-code-blocks";
import {BashSnippet} from "../snippets/BashSnippet";
import Image1 from "../../../../static/dog-breed-classification-1.png";
import Image2 from "../../../../static/dog-breed-classification-2.svg";
import Image3 from "../../../../static/impressed.gif";
import Image4 from "../../../../static/dog-breed-classification-3.png";
import Image5 from "../../../../static/dog-breed-classification-4.png";
import {lossValues} from "../data/DogBreedClassficationData";

const {Title, Paragraph} = Typography;
const {Panel} = Collapse;


class Chart extends PureComponent {
    render() {
        return (
            <ResponsiveContainer width="70%" height={450}>
                <LineChart
                    width={500}
                    height={450}
                    data={lossValues}
                    margin={{
                        top: 5,
                        right: 30,
                        left: 50,
                        bottom: 50,
                    }}
                    style={{"backgroundColor": "white"}}
                >
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis dataKey="epochs">
                        <Label value="Epochs" position="bottom" offset={10}/>
                    </XAxis>
                    <YAxis>
                        <Label angle={270} value="Loss Value" position="left" offset={10}/>
                    </YAxis>
                    <Tooltip/>
                    <Legend verticalAlign="top" height={36}/>
                    <Line type="monotone" dataKey="training" stroke="#8884d8" activeDot={{r: 8}}/>
                    <Line type="monotone" dataKey="validation" stroke="#82ca9d"/>
                </LineChart>
            </ResponsiveContainer>
        );
    }
}


class DogBreedClassification extends React.Component {
    render() {
        return (<>
            <Typography>
                <Paragraph>
                    In continuation to the <Link to={"/posts/dog-detection"}>previous post</Link>, where I played around
                    with PyTorch to detect dog images, in this post, I will train a Convolutional Neural Network to
                    classify breeds of dogs using PyTorch. The images for the project can be downloaded from <a
                    href={"https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"}>here</a>.
                </Paragraph>
                <Title level={3}>Loading Data</Title>
                <Paragraph>
                    Our initial step before we start training is to define our data loaders. For this, we will be
                    using <a
                    href={"https://pytorch.org/docs/stable/data.html#module-torch.utils.data"}>PyTorch DataLoader</a>.
                    We need to provide the path for the training, validation and testing images. As part of the
                    training, we will need to apply some transformations to the data such as resizing, random flips,
                    rotations, normalization, etc. We can apply the transformation as part of the data loader so that
                    transformations are already handled when images are fed into the network. Images can be trained in
                    batches, so we will also provide the batch size for when they are fed to the model.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"transform_train = transforms.Compose([\n" +
            "    transforms.Resize(256),\n" +
            "    transforms.RandomHorizontalFlip(),\n" +
            "    transforms.RandomRotation(10),\n" +
            "    transforms.CenterCrop(224),\n" +
            "    transforms.ToTensor(),\n" +
            "    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])\n" +
            "])\n" +
            "\n" +
            "train_loader = DataLoader(\n" +
            "    datasets.ImageFolder(\n" +
            "        \"/data/dog_images/train\",\n" +
            "        transform=transform_train),\n" +
            "    shuffle=True,\n" +
            "    batch_size=64,\n" +
            "    num_workers=6)"}/>
            <Typography>
                <Paragraph>
                    I defined a class that provides access to train, test and validation datasets as below:
                </Paragraph>
            </Typography>
            <Collapse accordian>
                <Panel header="Data Provider" key="1">
                    <PythonSnippet text={"class DataProvider:\n" +
                    "    def __init__(self, root_dir: str, **kwargs):\n" +
                    "        self._root_dir = root_dir\n" +
                    "        self._train_subfolder = kwargs.pop(\"train_subfolder\", \"train\")\n" +
                    "        self._test_subfolder = kwargs.pop(\"test_subfolder\", \"test\")\n" +
                    "        self._validation_subfolder = kwargs.pop(\"validation_subfolder\", \"valid\")\n" +
                    "        self._batch_size = kwargs.pop(\"batch_size\", 64)\n" +
                    "        self._num_workers = kwargs.pop(\"num_workers\", 0)\n" +
                    "\n" +
                    "        logger.info(f\"ROOT_DIR: {self._root_dir}\")\n" +
                    "        logger.info(f\"BATCH_SIZE: {self._batch_size}\")\n" +
                    "        logger.info(f\"NUM WORKERS: {self._num_workers}\")\n" +
                    "\n" +
                    "        transform_train = transforms.Compose([\n" +
                    "            transforms.Resize(256),\n" +
                    "            transforms.RandomHorizontalFlip(),\n" +
                    "            transforms.RandomRotation(10),\n" +
                    "            transforms.CenterCrop(224),\n" +
                    "            transforms.ToTensor(),\n" +
                    "            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])\n" +
                    "        ])\n" +
                    "\n" +
                    "        transform_others = transforms.Compose([\n" +
                    "            transforms.Resize(256),\n" +
                    "            transforms.CenterCrop(224),\n" +
                    "            transforms.ToTensor(),\n" +
                    "            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])\n" +
                    "        ])\n" +
                    "\n" +
                    "        self._train_loader = DataLoader(\n" +
                    "            datasets.ImageFolder(\n" +
                    "                os.path.join(root_dir, self._train_subfolder),\n" +
                    "                transform=transform_train),\n" +
                    "            shuffle=True,\n" +
                    "            batch_size=self._batch_size,\n" +
                    "            num_workers=self._num_workers)\n" +
                    "\n" +
                    "        self._validation_loader = DataLoader(\n" +
                    "            datasets.ImageFolder(\n" +
                    "                os.path.join(root_dir, self._validation_subfolder),\n" +
                    "                transform=transform_others),\n" +
                    "            shuffle=True,\n" +
                    "            batch_size=self._batch_size,\n" +
                    "            num_workers=self._num_workers)\n" +
                    "\n" +
                    "        self._test_loader = DataLoader(\n" +
                    "            datasets.ImageFolder(os.path.join(root_dir, self._test_subfolder),\n" +
                    "                                 transform=transform_others),\n" +
                    "            shuffle=False,\n" +
                    "            batch_size=self._batch_size,\n" +
                    "            num_workers=self._num_workers)\n" +
                    "\n" +
                    "    @property\n" +
                    "    def train(self) -> DataLoader:\n" +
                    "        return self._train_loader\n" +
                    "\n" +
                    "    @property\n" +
                    "    def test(self) -> DataLoader:\n" +
                    "        return self._test_loader\n" +
                    "\n" +
                    "    @property\n" +
                    "    def validation(self) -> DataLoader:\n" +
                    "        return self._validation_loader"}/>
                </Panel>
            </Collapse>
            <br/>
            <Typography>
                <Title level={3}>Architecture</Title>
                <Paragraph>
                    This is a practical post, so I won't get into too much detail on how I came up with the
                    architecture. However, the general idea is to apply convolutional filters to the image to capture
                    spatial features from the image and then use pooling to enhance the features after each layer. I
                    will be using 5 convolutional layers, with each convolutional layer followed by a max-pooling layer.
                    The convolutional
                    layers are followed by 2 fully-connected layers with ReLU activation and dropout in the middle.
                </Paragraph>
                <Paragraph>
                    I defined the architecture using a <Code text={"nn.Module"} language={"python"}
                                                             theme={dracula}/> implementation.
                </Paragraph>
            </Typography>
            <Collapse defaultActiveKey={['1']} accordian>
                <Panel header="Neural Net Architecture" key="1">
                    <PythonSnippet text={"class NeuralNet(nn.Module):\n" +
                    "    def __init__(self):\n" +
                    "        super(NeuralNet, self).__init__()\n" +
                    "\n" +
                    "        self.conv1 = nn.Conv2d(3, 16, (3, 3))\n" +
                    "        self.pool1 = nn.MaxPool2d(2, 2)\n" +
                    "\n" +
                    "        self.conv2 = nn.Conv2d(16, 32, (3, 3))\n" +
                    "        self.pool2 = nn.MaxPool2d(2, 2)\n" +
                    "\n" +
                    "        self.conv3 = nn.Conv2d(32, 64, (3, 3))\n" +
                    "        self.pool3 = nn.MaxPool2d(2, 2)\n" +
                    "\n" +
                    "        self.conv4 = nn.Conv2d(64, 128, (3, 3))\n" +
                    "        self.pool4 = nn.MaxPool2d(2, 2)\n" +
                    "\n" +
                    "        self.conv5 = nn.Conv2d(128, 256, (3, 3))\n" +
                    "        self.pool5 = nn.MaxPool2d(2, 2)\n" +
                    "\n" +
                    "        self.fc1 = nn.Linear(5 * 5 * 256, 400)\n" +
                    "        self.dropout = nn.Dropout(0.3)\n" +
                    "        self.fc2 = nn.Linear(400, 133)\n" +
                    "\n" +
                    "    def forward(self, x):\n" +
                    "        x = self.pool1(functional.relu(self.conv1(x)))\n" +
                    "        x = self.pool2(functional.relu(self.conv2(x)))\n" +
                    "        x = self.pool3(functional.relu(self.conv3(x)))\n" +
                    "        x = self.pool4(functional.relu(self.conv4(x)))\n" +
                    "        x = self.pool5(functional.relu(self.conv5(x)))\n" +
                    "\n" +
                    "        x = x.view(-1, 5 * 5 * 256)\n" +
                    "        x = functional.relu(self.fc1(x))\n" +
                    "        x = self.dropout(x)\n" +
                    "        x = self.fc2(x)\n" +
                    "        return x"}/>
                </Panel>
            </Collapse>
            <img
                alt="Architecture"
                src={Image5}
                style={{
                    width: "75%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }}
            />
            <br/>
            <Typography>
                <Title level={3}>Model Training and Validation</Title>
                <Paragraph>
                    Now that we have our data loader and model defined, the next step is to create the training and
                    validation routine. I will be doing the training and validation in a single step, checking the
                    validation loss at each epoch to check whether we have reduced validation loss.
                </Paragraph>
                <Paragraph>
                    During the training phase, you will have to activate training mode in PyTorch by calling
                    <Code text={"model.train()"} language={"python"} theme={dracula}/> on the model instance. After
                    that, we need to follow the following steps:
                </Paragraph>
                <ul>
                    <li>
                        Zero out optimizer gradients
                    </li>
                    <li>
                        Apply forward pass or <Code text={"model(input)"} language={"python"} theme={dracula}/>
                    </li>
                    <li>
                        Calculate the loss function
                    </li>
                    <li>
                        Apply the backward pass or <Code text={"loss.backward()"} language={"python"} theme={dracula}/>
                    </li>
                    <li>
                        Take new optimization step
                    </li>
                </ul>
                <Paragraph>
                    I am using Cross Entropy Loss function and Adam optimizer for training the ConvNet. The steps look
                    like below in code:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"neural_net.train()\n" +
            "for batch_index, (data, target) in enumerate(\n" +
            "        self._data_provider.train):\n" +
            "    logger.debug(f\"[TRAIN] Processing Batch: {batch_index}\")\n" +
            "    if self._use_gpu:\n" +
            "        data, target = data.cuda(), target.cuda()\n" +
            "    optimizer.zero_grad()\n" +
            "    output = neural_net(data)\n" +
            "    loss = self._criterion(output, target)\n" +
            "    loss.backward()\n" +
            "    optimizer.step()\n" +
            "    train_loss = train_loss + (\n" +
            "            (loss.item() - train_loss) / (batch_index + 1))"}/>
            <Typography>
                <Paragraph>
                    As we are training batches, each iteration will pass multiple images to the training loop. I am
                    using a batch size of 64 hence the input set of images would look like:
                </Paragraph>
                <img
                    alt="Dog images"
                    src={Image1}
                    style={{
                        width: "50%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Paragraph>
                    Validation step looks similar but simpler, where we dont need to calculate any gradients and just
                    calculate loss function from the forward pass.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"neural_net.eval()\n" +
            "for batch_index, (data, target) in enumerate(\n" +
            "        self._data_provider.validation):\n" +
            "    logger.debug(f\"[VALIDATE] Processing Batch: {batch_index}\")\n" +
            "    if self._use_gpu:\n" +
            "        data, target = data.cuda(), target.cuda()\n" +
            "\n" +
            "    with torch.no_grad():\n" +
            "        output = neural_net(data)\n" +
            "    loss = self._criterion(output, target)\n" +
            "    validation_loss = validation_loss + (\n" +
            "            (loss.item() - validation_loss) / (batch_index + 1))"}/>
            <Typography>
                <Paragraph>
                    One thing to note is that if you are training on a GPU, you need to move your data and model to the
                    GPU by calling <Code text={".cuda()"} language={"python"} theme={dracula}/> assuming training on
                    NVidia GPUs.
                </Paragraph>
                <Paragraph>
                    I save the model, everytime the validation loss decreases. So, we will automatically have the model
                    with the minimum validation loss by the end of the training loop.
                </Paragraph>
            </Typography>
            <Collapse accordian>
                <Panel header="Training / Validation Logs" key="1">
                    <BashSnippet
                        text={"2019-03-02 21:57:47.630 | INFO     | __main__:__init__:28 - ROOT_DIR: /data/dog_images/\n" +
                        "2019-03-02 21:57:47.630 | INFO     | __main__:__init__:29 - BATCH_SIZE: 64\n" +
                        "2019-03-02 21:57:47.630 | INFO     | __main__:__init__:30 - NUM WORKERS: 0\n" +
                        "2019-03-02 21:57:47.673 | INFO     | __main__:__init__:138 - CUDA is enabled - using GPU\n" +
                        "2019-03-02 21:57:49.294 | INFO     | __main__:train:147 - Model Architecture: \n" +
                        "NeuralNet(\n" +
                        "  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))\n" +
                        "  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                        "  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))\n" +
                        "  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                        "  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))\n" +
                        "  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                        "  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))\n" +
                        "  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                        "  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))\n" +
                        "  (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                        "  (fc1): Linear(in_features=6400, out_features=400, bias=True)\n" +
                        "  (dropout): Dropout(p=0.3, inplace=False)\n" +
                        "  (fc2): Linear(in_features=400, out_features=133, bias=True)\n" +
                        ")\n" +
                        "2019-03-02 21:57:49.294 | INFO     | __main__:_train_epoch:175 - [Epoch 0] Starting training phase\n" +
                        "2019-03-02 21:58:52.011 | INFO     | __main__:_train_epoch:189 - [Epoch 0] Starting eval phase\n" +
                        "2019-03-02 21:58:58.073 | INFO     | __main__:train:159 - Validation Loss Decreased: inf => 4.704965. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 21:58:58.096 | INFO     | __main__:_train_epoch:175 - [Epoch 1] Starting training phase\n" +
                        "2019-03-02 22:00:01.268 | INFO     | __main__:_train_epoch:189 - [Epoch 1] Starting eval phase\n" +
                        "2019-03-02 22:00:07.681 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.704965 => 4.440237. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:00:07.700 | INFO     | __main__:_train_epoch:175 - [Epoch 2] Starting training phase\n" +
                        "2019-03-02 22:01:10.125 | INFO     | __main__:_train_epoch:189 - [Epoch 2] Starting eval phase\n" +
                        "2019-03-02 22:01:16.491 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.440237 => 4.318264. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:01:16.514 | INFO     | __main__:_train_epoch:175 - [Epoch 3] Starting training phase\n" +
                        "2019-03-02 22:02:19.505 | INFO     | __main__:_train_epoch:189 - [Epoch 3] Starting eval phase\n" +
                        "2019-03-02 22:02:25.860 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.318264 => 4.110924. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:02:25.886 | INFO     | __main__:_train_epoch:175 - [Epoch 4] Starting training phase\n" +
                        "2019-03-02 22:03:27.334 | INFO     | __main__:_train_epoch:189 - [Epoch 4] Starting eval phase\n" +
                        "2019-03-02 22:03:33.551 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.110924 => 4.011838. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:03:33.574 | INFO     | __main__:_train_epoch:175 - [Epoch 5] Starting training phase\n" +
                        "2019-03-02 22:04:37.324 | INFO     | __main__:_train_epoch:189 - [Epoch 5] Starting eval phase\n" +
                        "2019-03-02 22:04:43.689 | INFO     | __main__:train:159 - Validation Loss Decreased: 4.011838 => 3.990763. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:04:43.718 | INFO     | __main__:_train_epoch:175 - [Epoch 6] Starting training phase\n" +
                        "2019-03-02 22:05:46.730 | INFO     | __main__:_train_epoch:189 - [Epoch 6] Starting eval phase\n" +
                        "2019-03-02 22:05:53.396 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.990763 => 3.849251. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:05:53.421 | INFO     | __main__:_train_epoch:175 - [Epoch 7] Starting training phase\n" +
                        "2019-03-02 22:06:58.002 | INFO     | __main__:_train_epoch:189 - [Epoch 7] Starting eval phase\n" +
                        "2019-03-02 22:07:04.432 | INFO     | __main__:_train_epoch:175 - [Epoch 8] Starting training phase\n" +
                        "2019-03-02 22:08:07.631 | INFO     | __main__:_train_epoch:189 - [Epoch 8] Starting eval phase\n" +
                        "2019-03-02 22:08:14.045 | INFO     | __main__:_train_epoch:175 - [Epoch 9] Starting training phase\n" +
                        "2019-03-02 22:09:16.695 | INFO     | __main__:_train_epoch:189 - [Epoch 9] Starting eval phase\n" +
                        "2019-03-02 22:09:23.421 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.849251 => 3.717872. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:09:23.445 | INFO     | __main__:_train_epoch:175 - [Epoch 10] Starting training phase\n" +
                        "2019-03-02 22:10:27.581 | INFO     | __main__:_train_epoch:189 - [Epoch 10] Starting eval phase\n" +
                        "2019-03-02 22:10:34.121 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.717872 => 3.588202. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:10:34.144 | INFO     | __main__:_train_epoch:175 - [Epoch 11] Starting training phase\n" +
                        "2019-03-02 22:11:38.351 | INFO     | __main__:_train_epoch:189 - [Epoch 11] Starting eval phase\n" +
                        "2019-03-02 22:11:44.652 | INFO     | __main__:_train_epoch:175 - [Epoch 12] Starting training phase\n" +
                        "2019-03-02 22:12:47.946 | INFO     | __main__:_train_epoch:189 - [Epoch 12] Starting eval phase\n" +
                        "2019-03-02 22:12:54.915 | INFO     | __main__:_train_epoch:175 - [Epoch 13] Starting training phase\n" +
                        "2019-03-02 22:13:58.543 | INFO     | __main__:_train_epoch:189 - [Epoch 13] Starting eval phase\n" +
                        "2019-03-02 22:14:04.912 | INFO     | __main__:_train_epoch:175 - [Epoch 14] Starting training phase\n" +
                        "2019-03-02 22:15:07.638 | INFO     | __main__:_train_epoch:189 - [Epoch 14] Starting eval phase\n" +
                        "2019-03-02 22:15:14.058 | INFO     | __main__:_train_epoch:175 - [Epoch 15] Starting training phase\n" +
                        "2019-03-02 22:16:17.191 | INFO     | __main__:_train_epoch:189 - [Epoch 15] Starting eval phase\n" +
                        "2019-03-02 22:16:23.634 | INFO     | __main__:_train_epoch:175 - [Epoch 16] Starting training phase\n" +
                        "2019-03-02 22:17:26.982 | INFO     | __main__:_train_epoch:189 - [Epoch 16] Starting eval phase\n" +
                        "2019-03-02 22:17:33.304 | INFO     | __main__:_train_epoch:175 - [Epoch 17] Starting training phase\n" +
                        "2019-03-02 22:18:36.207 | INFO     | __main__:_train_epoch:189 - [Epoch 17] Starting eval phase\n" +
                        "2019-03-02 22:18:42.790 | INFO     | __main__:_train_epoch:175 - [Epoch 18] Starting training phase\n" +
                        "2019-03-02 22:19:46.360 | INFO     | __main__:_train_epoch:189 - [Epoch 18] Starting eval phase\n" +
                        "2019-03-02 22:19:53.077 | INFO     | __main__:train:159 - Validation Loss Decreased: 3.588202 => 3.558330. Saving Model to /home/ksharma/tmp/dog_breed_classifier.model\n" +
                        "2019-03-02 22:19:53.106 | INFO     | __main__:_train_epoch:175 - [Epoch 19] Starting training phase\n" +
                        "2019-03-02 22:20:57.718 | INFO     | __main__:_train_epoch:189 - [Epoch 19] Starting eval phase\n" +
                        "2019-03-02 22:21:04.343 | INFO     | __main__:train:248 - Training Results: TrainedModel(train_losses=[4.828373968033565, 4.561895974477133, 4.3049539429800845, 4.10613343602135, 3.9616453170776356, 3.837490134012132, 3.729485934121267, 3.6096336637224464, 3.4845925603594092, 3.390888084684101, 3.2799783706665036, 3.20562988917033, 3.072563396181379, 2.9623924732208247, 2.870406493686495, 2.7523970808301663, 2.665678980236962, 2.535139397212437, 2.430639664332072, 2.333072783833458], validation_losses=[4.704964978354318, 4.440237283706664, 4.318263803209577, 4.110924073628017, 4.011837703841073, 3.990762727601187, 3.8492509978158136, 3.8797887223107472, 3.911121691976275, 3.717871563775199, 3.5882019826344083, 3.6028132949556624, 3.6062802246638705, 3.741273845945086, 3.6166011095047, 3.5896864277975893, 3.968828797340393, 3.668894120625087, 3.558329514094762, 3.6221354859215875], optimal_validation_loss=3.558329514094762)\n" +
                        "Process finished with exit code 0\n"} hideLineNumbers/>
                </Panel>
            </Collapse>
            <br/>
            <Typography>
                <Paragraph>
                    Let's look at the loss function for the training and validation routines.
                </Paragraph>
            </Typography>
            <Chart style={{
                width: "50%",
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
            }}/>
            <br/>
            <Alert
                message="NOTE"
                description="The validation loss stops improving much after epoch 10. Training loss keeps decreasing, as expected"
                type="info"
                showIcon
            />
            <br/>
            <Typography>
                <Title level={3}>Testing</Title>
                <Paragraph>
                    During the testing phase, I load the model that was saved earlier during the training phase to run
                    it over test images. We check the final activation of each image and apply the label based on the
                    category with max activation.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"model.eval()\n" +
            "for batch_idx, (data, target) in enumerate(self._data_provider.test):\n" +
            "    if self._use_gpu:\n" +
            "        data, target = data.cuda(), target.cuda()\n" +
            "    output = model(data)\n" +
            "    loss = self._criterion(output, target)\n" +
            "    test_loss = test_loss + (\n" +
            "            (loss.data.item() - test_loss) / (batch_idx + 1))\n" +
            "    predicted = output.max(1).indices\n" +
            "    predicted_labels = np.append(predicted_labels, predicted.numpy())\n" +
            "    target_labels = np.append(target_labels, target.numpy())"}/>
            <BashSnippet
                text={"2019-03-03 21:54:56.620 | INFO     | breed_classifier:__init__:27 - ROOT_DIR: /data/dog_images/\n" +
                "2019-03-03 21:54:56.620 | INFO     | breed_classifier:__init__:28 - BATCH_SIZE: 64\n" +
                "2019-03-03 21:54:56.620 | INFO     | breed_classifier:__init__:29 - NUM WORKERS: 0\n" +
                "2019-03-03 21:54:56.663 | INFO     | breed_classifier:__init__:137 - CUDA is enabled - using GPU\n" +
                "2019-03-03 21:55:04.558 | INFO     | __main__:test:39 - Test Results: TestResult(test_loss=3.607608267239162, correct_labels=161, total_labels=836)"}
                hideLineNumbers/>
            <br/>
            <Typography>
                <Paragraph>
                    Let's look at some of the prediction results in the images below. <span
                    style={{"color": "red"}}>Red</span> and <span style={{"color": "green"}}>Green</span> boxes indicate
                    incorrect and correct predictions, respectively.
                </Paragraph>
            </Typography>
            <img
                alt="Class9fication Test Results"
                src={Image4}
                style={{
                    width: "80%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }}
            />
            <br/>
            <img
                alt="Impressed"
                src={Image3}
                style={{
                    width: "20%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }}
            />
            <br/>
            <Alert
                message="Result"
                description="The final test accuracy is around 19%, which is... well not that good but... not that bad either considering we didn't use a very deep CNN. The final test loss value is close to our final validation loss value"
                type="info"
                showIcon
            />
            <br/>
            <Alert
                message="Spoiler Alert"
                description={"We can do much better than this using Transfer Learning on a pre-trained model like VGG-16, that we used to detect dog images in the precious post"}
                type="warning"
                showIcon
            />
            <br/>
            <Typography>
                <Paragraph>
                    The final Model class looks like below:
                </Paragraph>
            </Typography>
            <Collapse accordian>
                <Panel header="Model Training" key="1">
                    <PythonSnippet text={"TrainedModel = namedtuple(\n" +
                    "    \"TrainedModel\",\n" +
                    "    [\"train_losses\", \"validation_losses\", \"optimal_validation_loss\"])\n" +
                    "\n" +
                    "TestResult = namedtuple(\n" +
                    "    \"TestResult\", [\"test_loss\", \"correct_labels\", \"total_labels\"])\n" +
                    "\n" +
                    "\n" +
                    "class Model:\n" +
                    "    def __init__(self, data_provider: DataProvider,\n" +
                    "                 save_path: str, **kwargs):\n" +
                    "        self._data_provider = data_provider\n" +
                    "        self._save_path = save_path\n" +
                    "        self._criterion = nn.CrossEntropyLoss()\n" +
                    "        self._use_gpu = kwargs.pop(\"use_gpu\",\n" +
                    "                                   False) and torch.cuda.is_available()\n" +
                    "        if self._use_gpu:\n" +
                    "            logger.info(\"CUDA is enabled - using GPU\")\n" +
                    "        else:\n" +
                    "            logger.info(\"GPU Disabled: Using CPU\")\n" +
                    "\n" +
                    "    def train(self, n_epochs) -> TrainedModel:\n" +
                    "\n" +
                    "        neural_net = NeuralNet()\n" +
                    "        if self._use_gpu:\n" +
                    "            neural_net = neural_net.cuda()\n" +
                    "        logger.info(f\"Model Architecture: \\n{neural_net}\")\n" +
                    "        optimizer = optim.Adam(neural_net.parameters())\n" +
                    "        validation_losses = []\n" +
                    "        train_losses = []\n" +
                    "        min_validation_loss = np.Inf\n" +
                    "\n" +
                    "        for epoch in range(n_epochs):\n" +
                    "            train_loss, validation_loss = self._train_epoch(epoch, neural_net,\n" +
                    "                                                            optimizer)\n" +
                    "            validation_losses.append(validation_loss)\n" +
                    "            train_losses.append(train_loss)\n" +
                    "            if min_validation_loss > validation_loss:\n" +
                    "                logger.info(\n" +
                    "                    \"Validation Loss Decreased: {:.6f} => {:.6f}. \"\n" +
                    "                    \"Saving Model to {}\".format(\n" +
                    "                        min_validation_loss, validation_loss, self._save_path))\n" +
                    "                min_validation_loss = validation_loss\n" +
                    "                torch.save(neural_net.state_dict(), self._save_path)\n" +
                    "\n" +
                    "        optimal_model = NeuralNet()\n" +
                    "        optimal_model.load_state_dict(torch.load(self._save_path))\n" +
                    "        return TrainedModel(train_losses=train_losses,\n" +
                    "                            validation_losses=validation_losses,\n" +
                    "                            optimal_validation_loss=min_validation_loss)\n" +
                    "\n" +
                    "    def _train_epoch(self, epoch: int, neural_net: nn.Module,\n" +
                    "                     optimizer: optim.Optimizer):\n" +
                    "        train_loss = 0\n" +
                    "        logger.info(f\"[Epoch {epoch}] Starting training phase\")\n" +
                    "        neural_net.train()\n" +
                    "        for batch_index, (data, target) in enumerate(\n" +
                    "                self._data_provider.train):\n" +
                    "            if self._use_gpu:\n" +
                    "                data, target = data.cuda(), target.cuda()\n" +
                    "            optimizer.zero_grad()\n" +
                    "            output = neural_net(data)\n" +
                    "            loss = self._criterion(output, target)\n" +
                    "            loss.backward()\n" +
                    "            optimizer.step()\n" +
                    "            train_loss = train_loss + (\n" +
                    "                    (loss.item() - train_loss) / (batch_index + 1))\n" +
                    "\n" +
                    "        logger.info(f\"[Epoch {epoch}] Starting eval phase\")\n" +
                    "\n" +
                    "        validation_loss = 0\n" +
                    "        neural_net.eval()\n" +
                    "        for batch_index, (data, target) in enumerate(\n" +
                    "                self._data_provider.validation):\n" +
                    "            if self._use_gpu:\n" +
                    "                data, target = data.cuda(), target.cuda()\n" +
                    "            with torch.no_grad():\n" +
                    "                output = neural_net(data)\n" +
                    "            loss = self._criterion(output, target)\n" +
                    "            validation_loss = validation_loss + (\n" +
                    "                    (loss.item() - validation_loss) / (batch_index + 1))\n" +
                    "\n" +
                    "        return train_loss, validation_loss\n" +
                    "\n" +
                    "    def test(self) -> TestResult:\n" +
                    "        model = NeuralNet()\n" +
                    "        model.load_state_dict(torch.load(self._save_path))\n" +
                    "        if self._use_gpu:\n" +
                    "            model = model.cuda()\n" +
                    "        test_loss = 0\n" +
                    "        predicted_labels = np.array([])\n" +
                    "        target_labels = np.array([])\n" +
                    "\n" +
                    "        model.eval()\n" +
                    "        for batch_idx, (data, target) in enumerate(self._data_provider.test):\n" +
                    "            if self._use_gpu:\n" +
                    "                data, target = data.cuda(), target.cuda()\n" +
                    "            output = model(data)\n" +
                    "            loss = self._criterion(output, target)\n" +
                    "            test_loss = test_loss + (\n" +
                    "                    (loss.data.item() - test_loss) / (batch_idx + 1))\n" +
                    "            predicted = output.max(1).indices\n" +
                    "            predicted_labels = np.append(predicted_labels, predicted.numpy())\n" +
                    "            target_labels = np.append(target_labels, target.numpy())\n" +
                    "\n" +
                    "        return TestResult(test_loss=test_loss,\n" +
                    "                          correct_labels=sum(np.equal(target_labels,\n" +
                    "                                                      predicted_labels)),\n" +
                    "                          total_labels=len(target_labels))"}/>
                </Panel>
            </Collapse>
            <br/>
            <Typography>
                <Title level={3}>
                    Final Thoughts
                </Title>
                <Paragraph>
                    In this post, I trained a convolutional neural net from scratch to classify dog breeds. In a follow
                    up post, my plan is to use transfer learning to significantly improve the accuracy of the model.
                </Paragraph>
            </Typography>
        </>);
    }
}

export default DogBreedClassification;
