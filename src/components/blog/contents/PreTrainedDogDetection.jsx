import React from "react";
import {Typography, Alert, Menu} from "antd";

import {PythonSnippet} from "../snippets/PythonSnippet";
import {BashSnippet} from "../snippets/BashSnippet";
import {Link} from "react-router-dom";
import WooHooGIF from "../../../../static/woohoo.gif";


const {Title, Paragraph} = Typography;


class PreTrainedDogDetection extends React.Component {
    render() {
        return (<>
            <Typography>
                <Paragraph>
                    In a <Link to={"/posts/playing-with-opencv"}>previous post</Link>, I played around with OpenCV to
                    detect human faces on images. In
                    this post, I
                    will do something similar with dog images. All the information related to files are in the previous
                    post. The images can be downloaded from <a
                    href={"https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"}>here</a>.
                </Paragraph>
                <Paragraph>
                    The goal of this post is to use <a href={"https://pytorch.org/"}>PyTorch</a> pretrained
                    deep-learning models to detect dogs in
                    images. In this post, I will be using pre-trained <a
                    href={"https://arxiv.org/abs/1409.1556"}>VGG-16</a> model on ImageNet. Same exercise can be done
                    with other
                    models trained on ImageNet such as Inception-v3, ResNet-50, etc. You can find all the PyTorch
                    pre-trained models <a href={"https://pytorch.org/vision/stable/models.html"}>here</a>.
                </Paragraph>
                <Title level={3}>Setup</Title>
                <Paragraph>Let's start with loading the pre-trained model. If the model does not exist in your cache,
                    PyTorch will download the model but you will only need to do that once unless you delete your model
                    cache. Loading a pre-trained model is a piece of cake - just load it from the module <strong
                        style={{"fontStyle": "italic"}}>torchvision.models</strong> as below:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"import torch\n" +
            "import torchvision.models as models\n" +
            "\n" +
            "pretrained_vgg16 = models.vgg16(pretrained=True)\n" +
            "print(pretrained_vgg16) # should print all the layers from the VGG-16 model"}/>
            <BashSnippet text={"VGG(\n" +
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
            "    (6): Linear(in_features=4096, out_features=1000, bias=True)\n" +
            "  )\n" +
            ")\n"} hideLineNumbers/>
            <br/>
            <Typography>
                <Title level={3}>Implementation</Title>
                <Paragraph>
                    Now, we will look at the implementation of the testing/prediction of images. To read the images, we
                    will continue to use OpenCV, as we used in the previous post. Since OpenCV loads images in BGR
                    format, we will also need to transform the images to RGB format, which PyTorch expects. Below is a
                    function to load the images:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"def read_image(image_path: str) -> np.ndarray:\n" +
            "    img = cv2.imread(image_path)\n" +
            "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n" +
            "    return img"}/>
            <Typography>
                <Paragraph>
                    To feed the images through the model, we would need to apply a few transformations to the image,
                    which was loaded as a numpy array:
                    <ol>
                        <li>
                            <a href={"https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor"}>ToTensor</a>:
                            We would need to transform the image into a PyTorch Tensor, which changes the
                            dimension so the numpy array to match what PyTorch expects i.e. […, H, W] shape.
                        </li>
                        <li>
                            <a href={"https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize"}>Resize</a>:
                            We will resize the image to 256x256 image.
                        </li>
                        <li>
                            <a href={"https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.CenterCrop"}>CenterCrop</a>:
                            We will grab the cropped center of the image of the given size
                        </li>
                        <li>
                            <a href={"https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize"}>Normalize</a>:
                            Finally, we will normalize the images using given mean and stds
                        </li>
                    </ol>
                    We can compose all these transformations as below:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"preprocess = transforms.Compose([\n" +
            "    transforms.ToTensor(),\n" +
            "    transforms.Resize(256),\n" +
            "    transforms.CenterCrop(224),\n" +
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406],\n" +
            "                         std=[0.229, 0.224, 0.225])\n" +
            "])"}/>
            <Typography>
                <Paragraph>
                    After transforming the image, we just need to pass the image through the model to get the loss
                    function. Our final predicted category will be the index with max loss. ImageNet categories 151-268
                    correspond to dogs so all we need to do is get those in predicted indices.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"with torch.no_grad(): # No backpropagation required since we are not training\n" +
            "    output = self._model(image)\n" +
            "    predicted = output.argmax()"}/>
            <Typography>
                <Paragraph>
                    Finally, I put everything into easy-to-use class, with cuda support as well to test images on GPU.
                </Paragraph>
            </Typography>
            <PythonSnippet text={"class DogDetector:\n" +
            "    IMAGENET_MIN_INDEX_DOG = 151\n" +
            "    IMAGENET_MAX_INDEX_DOG = 268\n" +
            "\n" +
            "    def __init__(self, use_gpu: bool = False):\n" +
            "        self._model = models.vgg16(pretrained=True)\n" +
            "        self._use_cuda = torch.cuda.is_available() and use_gpu\n" +
            "        if self._use_cuda:\n" +
            "            logger.info(\"CUDA is enabled - using GPU\")\n" +
            "            self._model = self._model.cuda()\n" +
            "\n" +
            "    @staticmethod\n" +
            "    def _read_image(image_path: str) -> np.ndarray:\n" +
            "        img = cv2.imread(image_path)\n" +
            "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n" +
            "        return img\n" +
            "\n" +
            "    def predict(self, image_path: str) -> int:\n" +
            "        image = self._read_image(image_path)\n" +
            "\n" +
            "        preprocess = transforms.Compose([\n" +
            "            transforms.ToTensor(),\n" +
            "            transforms.Resize(256),\n" +
            "            transforms.CenterCrop(224),\n" +
            "            transforms.Normalize(mean=[0.485, 0.456, 0.406],\n" +
            "                                 std=[0.229, 0.224, 0.225])\n" +
            "        ])\n" +
            "\n" +
            "        image = preprocess(image).unsqueeze_(0)\n" +
            "\n" +
            "        if self._use_cuda:\n" +
            "            image = image.cuda()\n" +
            "\n" +
            "        with torch.no_grad():\n" +
            "            output = self._model(image)\n" +
            "            predicted = output.argmax()\n" +
            "\n" +
            "        return predicted\n" +
            "\n" +
            "    @timeit\n" +
            "    def detect(self, image_path: str) -> bool:\n" +
            "        predicted_index = self.predict(image_path)\n" +
            "        logger.info(f\"Predicted Index: {predicted_index}\")\n" +
            "        return (self.IMAGENET_MIN_INDEX_DOG <= predicted_index <=\n" +
            "                self.IMAGENET_MAX_INDEX_DOG)\n"}/>
            <Typography>
                <Paragraph>
                    Let's test our DogDetector class on all the dog and human images:
                </Paragraph>
            </Typography>
            <PythonSnippet text={"dog_files = np.array(glob(\"/data/dog_images/*/*/*\"))\n" +
            "dog_detector = DogDetector(use_gpu=False)\n" +
            "chosen_size = 100\n" +
            "detected = sum(dog_detector.detect(f) for f in dog_files[:chosen_size])\n" +
            "logger.info(f\"Dogs detected in {detected} / {chosen_size} = \"\n" +
            "            f\"{detected * 100 / chosen_size}% images\")"}/>
            <BashSnippet text={"INFO     | __main__:<module>:6 - Dogs detected in 100 / 100 = 100.0% images"}
                         hideLineNumbers/>
            <br/>
            <PythonSnippet text={"human_files = np.array(glob(\"/data/lfw/*/*\"))\n" +
            "detected = sum(dog_detector.detect(f) for f in human_files[:chosen_size])\n" +
            "logger.info(f\"Dogs detected in {detected} / {chosen_size} = \"\n" +
            "            f\"{detected * 100 / chosen_size}% images\")"}/>
            <BashSnippet text={"INFO     | __main__:<module>:9 - Dogs detected in 0 / 100 = 0.0% images"}
                         hideLineNumbers/>
            <br/>
            <img src={WooHooGIF} alt="woohoo" style={
                {
                    width: "30%",
                    display: "block",
                    marginLeft: "auto",
                    marginRight: "auto",
                }
            }/>
            <br/>
            <Typography>
                <Title level={3}>Summary</Title>
                <Paragraph>
                    In this post, I used a pretrained VGG-16 model to test dog images to indicate whether images have
                    dogs or not. In a follow up project, I plan to extend this to use transfer learning to train a deep
                    learning model for predicting dog breeds.
                </Paragraph>
            </Typography>
        </>);
    }

}

export default PreTrainedDogDetection;