import React from "react";
import {Collapse} from 'antd';
import {Typography} from "antd";
import {Alert} from 'antd';
import {Link} from "react-router-dom";
import {PythonSnippet} from "../snippets/PythonSnippet";
import {Code, dracula} from "react-code-blocks";
import {BashSnippet} from "../snippets/BashSnippet";
import {CartesianGrid, Label, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import Image1 from "../../../../static/vgg-dog-breed-2.png";
import Image2 from "../../../../static/vgg-dog-breed-3.png";
import Image3 from "../../../../static/vgg-dog-breed-4.png";
import Image4 from "../../../../static/vgg-dog-breed-5.png";
import Image5 from "../../../../static/vgg-dog-breed-6.png";
import Image6 from "../../../../static/vgg-dog-breed-7.png";

const {Title, Paragraph} = Typography;
const {Panel} = Collapse;


const trainingLossData = [{'epochs': 1, 'training': 2.7716, 'validation': 1.222},
    {'epochs': 2, 'training': 2.0471, 'validation': 1.3032},
    {'epochs': 3, 'training': 2.0092, 'validation': 1.4058},
    {'epochs': 4, 'training': 1.9197, 'validation': 1.1522},
    {'epochs': 5, 'training': 2.0437, 'validation': 1.18},
    {'epochs': 6, 'training': 1.8903, 'validation': 1.0291}];


class VGGDogBreedClassification extends React.Component {
    render() {
        return (
            <>
                <Typography>
                    <Paragraph>
                        In a <Link to={"/posts/dog-breed-classification"}>previous post</Link>, I trained a neural net
                        from scratch to classify dog images using PyTorch. I
                        achieved ~20% accuracy on results, which isn't great. In this post, my goal is to significantly
                        improve on that accuracy by using <a href={"https://ruder.io/transfer-learning/"}>Transfer
                        Learning</a> on a pre-trained deeper neural network - <a
                        href={"https://arxiv.org/abs/1409.1556"}>VGG-16</a>. The images for the analysis can be
                        downloaded from <a
                        href={"https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"}>here</a>.
                    </Paragraph>
                    <Paragraph>
                        Transfer learning should be really effective here in training a deep neural net with
                        comparatively
                        little data since it allows us to use a lot of the same
                        information about dog features that VGG-16 has already learned and apply them to a new but
                        similar
                        problem. In other words, for this problem we begin from a good starting point since the model
                        does not have to learn "What is a dog?" or "How is this dog
                        different from that dog?" from scratch.
                    </Paragraph>
                    <Title level={3}>Initialize Architecture</Title>
                    <Paragraph>
                        It is fairly simple to initialize a new architecture that will match our problem. We need to
                        load the pre-trained model and change the last few fully connected layers to get the model
                        output to
                        resemble our target classes. One interesting thing about it is that we don't need to run
                        backprop on majority of the network since it is already accepted as optimal. In this case, we
                        won't run backprop on any of the convolutional layers and majority of the fully connected
                        layers. This would make our training really efficient. Let's look at how we would setup the
                        network in code:
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def get_model_arch() -> nn.Module:\n" +
                "    vgg16 = models.vgg16(pretrained=True)\n" +
                "    for param in vgg16.features.parameters():\n" +
                "        param.requires_grad = False  # pre-trained - dont touch\n" +
                "    n_inputs_final_layer = vgg16.classifier[-1].in_features\n" +
                "    n_classes = len(self._data_provider.train.dataset.classes)\n" +
                "    # Replace the final layer\n" +
                "    final_layer = nn.Linear(n_inputs_final_layer, n_classes)\n" +
                "    vgg16.classifier[-1] = final_layer\n" +
                "    return vgg16"}/>
                <Alert
                    message="NOTE"
                    description="We are changing the final layer to resemble our target classes."
                    type="info"
                    showIcon
                />
                <br/>
                <Typography>
                    <Title level={3}>Training, Validation, Testing</Title>
                    <Paragraph>
                        If you followed my <Link to={"/posts/dog-breed-classification"}>previous post</Link>, I defined
                        python classes for data loading, and model training. Since, most of the code looked the same, I
                        refactored it a little bit to handle both 1) training from scratch and 2) training using
                        transfer learning. We define the <Code text={"BaseModel"} theme={dracula}
                                                               language={"python"}/> class that handles all of the
                        training, validation and testing code. The derived classes just need to provide properties <Code
                        text={"train_model"} theme={dracula} language={"python"}/> and <Code text={"test_model"}
                                                                                             theme={dracula}
                                                                                             language={"python"}/>.
                    </Paragraph>
                </Typography>
                <Collapse accordian>
                    <Panel header="BaseModel" key="1">
                        <PythonSnippet text={"class BaseModel(metaclass=abc.ABCMeta):\n" +
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
                        "        self._verbose = kwargs.pop(\"verbose\", False)\n" +
                        "\n" +
                        "    @property\n" +
                        "    @abstractmethod\n" +
                        "    def train_model(self) -> nn.Module:\n" +
                        "        raise NotImplementedError(\"Implement in derived class\")\n" +
                        "\n" +
                        "    @property\n" +
                        "    @abstractmethod\n" +
                        "    def test_model(self) -> nn.Module:\n" +
                        "        raise NotImplementedError(\"Implement in base class\")\n" +
                        "\n" +
                        "    def train(self, n_epochs: int) -> TrainedModel:\n" +
                        "        model = self.train_model\n" +
                        "        optimizer = optim.Adam(model.parameters())\n" +
                        "        logger.info(f\"Model Architecture: \\n{model}\")\n" +
                        "\n" +
                        "        validation_losses = []\n" +
                        "        train_losses = []\n" +
                        "        min_validation_loss = np.Inf\n" +
                        "\n" +
                        "        for epoch in range(n_epochs):\n" +
                        "            train_loss, validation_loss = self._train_epoch(epoch, model,\n" +
                        "                                                            optimizer)\n" +
                        "            validation_losses.append(validation_loss)\n" +
                        "            train_losses.append(train_loss)\n" +
                        "            if min_validation_loss > validation_loss:\n" +
                        "                logger.info(\n" +
                        "                    \"Validation Loss Decreased: {:.6f} => {:.6f}. \"\n" +
                        "                    \"Saving Model to {}\".format(\n" +
                        "                        min_validation_loss, validation_loss, self._save_path))\n" +
                        "                min_validation_loss = validation_loss\n" +
                        "                torch.save(model.state_dict(), self._save_path)\n" +
                        "\n" +
                        "        return TrainedModel(train_losses=train_losses,\n" +
                        "                            validation_losses=validation_losses,\n" +
                        "                            optimal_validation_loss=min_validation_loss)\n" +
                        "\n" +
                        "    def _train_epoch(self, epoch: int, neural_net: nn.Module,\n" +
                        "                     optimizer: optim.Optimizer) -> Tuple[float, float]:\n" +
                        "        train_loss = 0\n" +
                        "        logger.info(f\"[Epoch {epoch}] Starting training phase\")\n" +
                        "        neural_net.train()\n" +
                        "        total_samples = len(self._data_provider.train.dataset.samples)\n" +
                        "        batch_count = (total_samples // self._data_provider.train.batch_size)\n" +
                        "        for batch_index, (data, target) in tqdm(enumerate(\n" +
                        "                self._data_provider.train), total=batch_count + 1, ncols=80):\n" +
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
                        "        total_samples = len(self._data_provider.validation.dataset.samples)\n" +
                        "        batch_count = (\n" +
                        "                total_samples // self._data_provider.validation.batch_size)\n" +
                        "        neural_net.eval()\n" +
                        "        for batch_index, (data, target) in tqdm(enumerate(\n" +
                        "                self._data_provider.validation), total=batch_count + 1,\n" +
                        "                ncols=80):\n" +
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
                        "        model = self.test_model\n" +
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
                        "            predicted_labels = np.append(predicted_labels,\n" +
                        "                                         predicted.cpu().numpy())\n" +
                        "            target_labels = np.append(target_labels, target.cpu().numpy())\n" +
                        "\n" +
                        "        return TestResult(test_loss=test_loss,\n" +
                        "                          correct_labels=sum(np.equal(target_labels,\n" +
                        "                                                      predicted_labels)),\n" +
                        "                          total_labels=len(target_labels))"}/>
                    </Panel>
                </Collapse>
                <br/>
                <Typography>
                    <Paragraph>
                        There are a few more differences in the hyper-parameters of transfer learning model. The
                        normalization means and stds in the new model are: norm_means = [0.485, 0.456, 0.406] and
                        norm_stds = [0.229, 0.224, 0.225]. We are also going to train it for fewer epochs. I chose 6
                        epochs for my analysis. Below are the training logs for the model:
                    </Paragraph>
                </Typography>
                <Collapse accordian>
                    <Panel header="Training and Validation Logs" key="1">
                        <BashSnippet
                            text={"2019-04-04 19:38:03.509 | INFO     | breed_classifier:__init__:31 - ROOT_DIR: /data/dog_images/\n" +
                            "2019-04-04 19:38:03.509 | INFO     | breed_classifier:__init__:32 - BATCH_SIZE: 64\n" +
                            "2019-04-04 19:38:03.509 | INFO     | breed_classifier:__init__:33 - NUM WORKERS: 0\n" +
                            "2019-04-04 19:38:03.553 | INFO     | dog_classifier.breed_classifier:__init__:149 - CUDA is enabled - using GPU\n" +
                            "2019-04-04 19:38:06.049 | INFO     | dog_classifier.breed_classifier:train:168 - Model Architecture: \n" +
                            "VGG(\n" +
                            "  (features): Sequential(\n" +
                            "    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (1): ReLU(inplace=True)\n" +
                            "    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (3): ReLU(inplace=True)\n" +
                            "    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                            "    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (6): ReLU(inplace=True)\n" +
                            "    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (8): ReLU(inplace=True)\n" +
                            "    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                            "    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (11): ReLU(inplace=True)\n" +
                            "    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (13): ReLU(inplace=True)\n" +
                            "    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (15): ReLU(inplace=True)\n" +
                            "    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                            "    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (18): ReLU(inplace=True)\n" +
                            "    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (20): ReLU(inplace=True)\n" +
                            "    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (22): ReLU(inplace=True)\n" +
                            "    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                            "    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (25): ReLU(inplace=True)\n" +
                            "    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (27): ReLU(inplace=True)\n" +
                            "    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n" +
                            "    (29): ReLU(inplace=True)\n" +
                            "    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n" +
                            "  )\n" +
                            "  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))\n" +
                            "  (classifier): Sequential(\n" +
                            "    (0): Linear(in_features=25088, out_features=4096, bias=True)\n" +
                            "    (1): ReLU(inplace=True)\n" +
                            "    (2): Dropout(p=0.5, inplace=False)\n" +
                            "    (3): Linear(in_features=4096, out_features=4096, bias=True)\n" +
                            "    (4): ReLU(inplace=True)\n" +
                            "    (5): Dropout(p=0.5, inplace=False)\n" +
                            "    (6): Linear(in_features=4096, out_features=133, bias=True)\n" +
                            "  )\n" +
                            ")\n" +
                            "2019-04-04 19:38:06.049 | INFO     | dog_classifier.breed_classifier:_train_epoch:194 - [Epoch 0] Starting training phase\n" +
                            "100%|█████████████████████████████████████████| 105/105 [01:19<00:00,  1.32it/s]\n" +
                            "2019-04-04 19:39:25.833 | INFO     | dog_classifier.breed_classifier:_train_epoch:210 - [Epoch 0] Starting eval phase\n" +
                            "100%|███████████████████████████████████████████| 14/14 [00:08<00:00,  1.62it/s]\n" +
                            "2019-04-04 19:39:34.487 | INFO     | dog_classifier.breed_classifier:train:180 - Validation Loss Decreased: inf => 1.221989. Saving Model to /home/ksharma/tmp/dog_breed_classifier_transfer.model\n" +
                            "2019-04-04 19:39:35.638 | INFO     | dog_classifier.breed_classifier:_train_epoch:194 - [Epoch 1] Starting training phase\n" +
                            "100%|█████████████████████████████████████████| 105/105 [01:20<00:00,  1.30it/s]\n" +
                            "2019-04-04 19:40:56.256 | INFO     | dog_classifier.breed_classifier:_train_epoch:210 - [Epoch 1] Starting eval phase\n" +
                            "100%|███████████████████████████████████████████| 14/14 [00:08<00:00,  1.59it/s]\n" +
                            "2019-04-04 19:41:05.066 | INFO     | dog_classifier.breed_classifier:_train_epoch:194 - [Epoch 2] Starting training phase\n" +
                            "100%|█████████████████████████████████████████| 105/105 [01:20<00:00,  1.31it/s]\n" +
                            "2019-04-04 19:42:25.420 | INFO     | dog_classifier.breed_classifier:_train_epoch:210 - [Epoch 2] Starting eval phase\n" +
                            "100%|███████████████████████████████████████████| 14/14 [00:08<00:00,  1.60it/s]\n" +
                            "2019-04-04 19:42:34.176 | INFO     | dog_classifier.breed_classifier:_train_epoch:194 - [Epoch 3] Starting training phase\n" +
                            "100%|█████████████████████████████████████████| 105/105 [01:21<00:00,  1.30it/s]\n" +
                            "2019-04-04 19:43:55.202 | INFO     | dog_classifier.breed_classifier:_train_epoch:210 - [Epoch 3] Starting eval phase\n" +
                            "100%|███████████████████████████████████████████| 14/14 [00:08<00:00,  1.60it/s]\n" +
                            "2019-04-04 19:44:03.979 | INFO     | dog_classifier.breed_classifier:train:180 - Validation Loss Decreased: 1.221989 => 1.152154. Saving Model to /home/ksharma/tmp/dog_breed_classifier_transfer.model\n" +
                            "2019-04-04 19:44:05.152 | INFO     | dog_classifier.breed_classifier:_train_epoch:194 - [Epoch 4] Starting training phase\n" +
                            "100%|█████████████████████████████████████████| 105/105 [01:20<00:00,  1.30it/s]\n" +
                            "2019-04-04 19:45:25.865 | INFO     | dog_classifier.breed_classifier:_train_epoch:210 - [Epoch 4] Starting eval phase\n" +
                            "100%|███████████████████████████████████████████| 14/14 [00:08<00:00,  1.61it/s]\n" +
                            "2019-04-04 19:45:34.572 | INFO     | dog_classifier.breed_classifier:_train_epoch:194 - [Epoch 5] Starting training phase\n" +
                            "100%|█████████████████████████████████████████| 105/105 [01:19<00:00,  1.32it/s]\n" +
                            "2019-04-04 19:46:54.352 | INFO     | dog_classifier.breed_classifier:_train_epoch:210 - [Epoch 5] Starting eval phase\n" +
                            "100%|███████████████████████████████████████████| 14/14 [00:08<00:00,  1.61it/s]\n" +
                            "2019-04-04 19:47:03.036 | INFO     | dog_classifier.breed_classifier:train:180 - Validation Loss Decreased: 1.152154 => 1.029065. Saving Model to /home/ksharma/tmp/dog_breed_classifier_transfer.model\n" +
                            "2019-04-04 19:47:04.199 | INFO     | __main__:train:31 - Training Results: TrainedModel(train_losses=[2.771625365529741, 2.0471301839465186, 2.0091793037596197, 1.9197123822711761, 2.043689425786336, 1.8903201120240343], validation_losses=[1.221989027091435, 1.3032183263983048, 1.405754255396979, 1.1521543094090054, 1.1800314315727778, 1.0290648672474032], optimal_validation_loss=1.0290648672474032)\n" +
                            "Process finished with exit code 0"} hideLineNumbers/>
                    </Panel>
                </Collapse>
                <br/>
                <ResponsiveContainer width="70%" height={450}>
                    <LineChart
                        width={"70%"}
                        height={450}
                        data={trainingLossData}
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
                        <XAxis dataKey="epochs">
                            <Label value="Epochs" position="bottom" offset={10} style={{color: "white"}}/>
                        </XAxis>
                        <YAxis>
                            <Label angle={270} value="Loss Value" position="left" offset={10}
                                   style={{"color": "white"}}/>
                        </YAxis>
                        <Tooltip/>
                        <Legend verticalAlign="top" height={36}/>
                        <Line type="monotone" dataKey="training" stroke="#8884d8" activeDot={{r: 8}}/>
                        <Line type="monotone" dataKey="validation" stroke="#82ca9d"/>
                    </LineChart>
                </ResponsiveContainer>
                <br/>
                <Typography>
                    <Paragraph>
                        Interestingly, we have a lower loss function value just after epoch 1 compared to the
                        original model I fit from scratch. Our final loss function is way lower than the loss function
                        from the original model. Let's look at the test results:
                    </Paragraph>
                </Typography>
                <Collapse defaultActiveKey={['1']} accordian>
                    <Panel header="Test Logs" key="1">
                        <BashSnippet
                            text={"2019-04-04 20:39:40.470 | INFO     | breed_classifier:__init__:31 - ROOT_DIR: /data/dog_images/\n" +
                            "2019-04-04 20:39:40.470 | INFO     | breed_classifier:__init__:32 - BATCH_SIZE: 64\n" +
                            "2019-04-04 20:39:40.470 | INFO     | breed_classifier:__init__:33 - NUM WORKERS: 0\n" +
                            "2019-04-04 20:39:40.509 | INFO     | dog_classifier.breed_classifier:__init__:149 - CUDA is enabled - using GPU\n" +
                            "2019-04-04 20:39:52.692 | INFO     | __main__:test:55 - Test Results: TestResult(test_loss=1.1680023499897547, correct_labels=564, total_labels=836)"}
                            hideLineNumbers/>
                    </Panel>
                </Collapse>
                <br/>
                <Alert
                    message="NOTE"
                    description="The test accuracy is almost 67%, way better than 19% from original model"
                    type="info"
                    showIcon
                />
                <br/>
                <Typography>
                    <Title level={3}>Results</Title>
                    <Paragraph>
                        Let's look at some of the results from our test images. Here is a random collection of 64 images
                        with <span
                        style={{"color": "red"}}>Red</span> and <span style={{"color": "green"}}>Green</span> boxes
                        indicating
                        incorrect and correct predictions, respectively:
                    </Paragraph>
                </Typography>
                <img
                    alt="Results"
                    src={Image1}
                    style={{
                        width: "75%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Typography>
                    <Paragraph>
                        Below are the worst performing dog breeds in the datasets with their corresponding test accuracy
                        rate.
                    </Paragraph>
                </Typography>
                <img
                    alt="Worst Accuracy"
                    src={Image2}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <img
                    alt="Worst-1"
                    src={Image3}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <img
                    alt="Worst-2"
                    src={Image4}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <img
                    alt="Worst-3"
                    src={Image5}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <img
                    alt="Worst-4"
                    src={Image6}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Alert
                    message="NOTE"
                    description="On the other hand, 30 breeds had a perfect 100% accuracy"
                    type="info"
                    showIcon
                />
                <br/>
                <Typography>
                    <Title level={3}>
                        Final Thoughts
                    </Title>
                    <Paragraph>
                        In this post, I applied transfer learning to VGG-16 model and used it for dog breed
                        classification. We achieved almost 67% accuracy rate on the test dataset compared to only 19%
                        with the original model from scratch.
                    </Paragraph>
                </Typography>
            </>
        );
    }
}

export default VGGDogBreedClassification;