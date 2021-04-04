import React from "react";
import {Typography} from "antd";

import {PythonSnippet} from "../snippets/PythonSnippet";
import {BashSnippet} from "../snippets/BashSnippet";
import Image1 from "../../../../static/face-detection-1.png";
import Image2 from "../../../../static/face-detection-2.png";
import Image3 from "../../../../static/face-detection-3.png";
import Image4 from "../../../../static/face-detection-4.png";

const {Title, Paragraph} = Typography;

class FaceDetection extends React.Component {
    render() {
        return (
            <div>
                <Typography>
                    <Paragraph>
                        I played around with OpenCV in python to experiment with face detection in images. In this post
                        I will cover:
                        <ul>
                            <li>
                                How to read image files into numpy array
                            </li>
                            <li>
                                Detect face in an image
                            </li>
                            <li>
                                Mark detected faces
                            </li>
                            <li>
                                Experiment on human and dog faces
                            </li>
                        </ul>
                    </Paragraph>
                    <Paragraph>
                        Firstly, let us download the images. The images can be downloaded from <a
                        href={"https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"}>Dog
                        Images</a> and <a href={"https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip"}>
                        Human Images
                    </a>. I downloaded the images, unzipped them, and put them in /data/ (my big second hard drive) on
                        my linux box. Depending on where you download images yourself, change the directory in the code
                        snippets below.
                    </Paragraph>
                    <Title level={3}>Loading Images</Title>
                    <Paragraph>
                        Loading images is very simple with OpenCV and images are loaded as numpy arrays
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"import cv2\ncv2.imread(file_name)"}/>
                <PythonSnippet text={
                    "file_name='/data/dog_images/train/124.Poodle/Poodle_07929.jpg'\nimage = cv2.imread(file_name)\nprint(image)"}/>
                <BashSnippet text={"array([[[135, 176, 155],\n" +
                "        [126, 167, 146],\n" +
                "        [107, 151, 128],\n" +
                "        ...,\n" +
                "        [ 72,  92, 109],\n" +
                "        [ 69,  89, 106],\n" +
                "        [ 65,  85, 102]]], dtype=uint8)"} hideLineNumbers/>
                <br/>
                <Typography>
                    <Paragraph>
                        You can also convert images into different color schemes. For example,
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"img = cv2.imread(file_name)\ngray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"}/>

                <Typography>
                    <Title level={3}>Face Detection</Title>
                    <Paragraph>
                        Now let's start using the tools that are part of OpenCV to detect faces in images. OpenCV ships
                        with the CascadeClassifier, which is an ensemble model used for image processing tasks such as
                        object detection and tracking, primarily facial detection and recognition
                    </Paragraph>
                    <Paragraph>
                        In the following snippet, we will initialize the cascade classifier and use it to detect faces.
                        If we detect any faces, we will mark a blue rectangle around the detected edges.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"DetectedFace = namedtuple(\"DetectedFace\", [\"faces\", \"image\"])"}/>
                <PythonSnippet text={"def detect_faces(file_name: str) -> DetectedFace:\n" +
                "    face_cascade = cv2.CascadeClassifier(\n" +
                "        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')\n" +
                "    img = cv2.imread(file_name)\n" +
                "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n" +
                "    faces = face_cascade.detectMultiScale(gray)\n" +
                "\n" +
                "    for (x, y, w, h) in faces:\n" +
                "        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)\n" +
                "\n" +
                "    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n" +
                "    return DetectedFace(faces=faces, image=cv_rgb)"}/>
                <Typography>
                    <Paragraph>
                        If we detect any faces, <strong>faces</strong> field in <strong>DetectedFace</strong> will have
                        non zero length. Below we define a function to check whether we detected any faces or not:
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def face_present(file_path: str) -> bool:\n" +
                "    img = detect_faces(file_path)\n" +
                "    return len(img.faces) > 0"}/>
                <Typography>
                    <Paragraph>
                        Finally, we define the function to plot the marked image.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def plot_detected_faces(img: DetectedFace):\n" +
                "    fig, ax = plt.subplots()\n" +
                "    ax.imshow(img.image)\n" +
                "    plt.show()"}/>
                <Typography>
                    <Paragraph>
                        Let's use the code above to test some sample images. We load up downloaded images into an
                        array and choose a random image to test our face detector. First, we try on a random human image
                        and then on a random dog image.
                    </Paragraph>
                    <Title level={4}>Human Images</Title>
                </Typography>
                <PythonSnippet text={"human_files = np.array(glob(\"/data/lfw/*/*\"))\n" +
                "dog_files = np.array(glob(\"/data/dog_images/*/*/*\"))\n" +
                "\n" +
                "image = detect_faces(human_files[np.random.randint(0, len(human_files))])\n" +
                "plot_detected_faces(image)"}/>
                <img
                    alt="human face detection"
                    src={Image1}
                    style={{
                        width: "30%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Typography>
                    <Title level={4}>Dog Images</Title>
                </Typography>
                <PythonSnippet text={"image = detect_faces(dog_files[np.random.randint(0, len(dog_files))])\n" +
                "plot_detected_faces(image)"}/>
                <img
                    alt="dog face detection"
                    src={Image2}
                    style={{
                        width: "30%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Typography>
                    <Paragraph>
                        Now, let's run the same code on a bunch of different images that we downloaded. I have selected
                        1000
                        images from both sets to run the face detector.
                    </Paragraph>
                </Typography>
                <PythonSnippet text={"def plot_detected_faces_multiple(results: List[DetectedFace],\n" +
                "                                 rows: int = 3, columns: int = 3):\n" +
                "    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))\n" +
                "    for r in range(rows):\n" +
                "        for c in range(columns):\n" +
                "            img = results[r * columns + c]\n" +
                "            ax[r][c].imshow(img.image)\n" +
                "    plt.show()"}/>
                <PythonSnippet text={"filter_count = 1000\n" +
                "\n" +
                "human_images_result = list(map(\n" +
                "    detect_faces,\n" +
                "    human_files[np.random.randint(0, len(human_files), filter_count)]))\n" +
                "dog_images_result = list(map(\n" +
                "    detect_faces,\n" +
                "    dog_files[np.random.randint(0, len(dog_files), filter_count)]))\n" +
                "\n" +
                "plot_detected_faces_multiple(human_images_result)\n" +
                "plot_detected_faces_multiple(dog_images_result)"}/>
                <img
                    alt="human face detection multiple"
                    src={Image4}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br/>
                <Typography>
                    <Paragraph>
                        As we can see, it does a great job at detecting human faces in all the images, even when there
                        are multiple humans in an image.
                    </Paragraph>
                    <Paragraph>
                        ...
                    </Paragraph>
                    <Paragraph>
                        On the dog images, not so much...
                    </Paragraph>
                </Typography>
                <img
                    alt="dog face detection multiple"
                    src={Image3}
                    style={{
                        width: "60%",
                        display: "block",
                        marginLeft: "auto",
                        marginRight: "auto",
                    }}
                />
                <br />
                <Typography>
                    <Title level={3}>Final Comments</Title>
                    <Paragraph>
                        This was a fun and small project to play around with OpenCV's image processing toolkit. My next goal is to have a better dog image classifier using CNN.
                    </Paragraph>
                </Typography>
            </div>
        );
    }
}

export default FaceDetection;