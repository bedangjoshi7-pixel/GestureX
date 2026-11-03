
# GestureX 

**Real-Time Hand Landmark Detection using Python, OpenCV and MediaPipe.**

GestureX is a computer vision project that detects and tracks hand landmarks in real time using a webcam.

I built this project while learning computer vision and exploring how machines can detect and understand hand movements through visual data.

## What is GestureX?

GestureX uses MediaPipe's Hand Landmarker to detect hands and identify 21 landmarks on each hand. These landmarks represent important points such as the wrist, finger joints and fingertips.

The project uses those coordinates to draw connections between the fingers and display the tracked hand in real time.

Currently, it supports detecting up to two hands simultaneously.

## Technologies Used

**Python**

The main programming language used to build the project.

**OpenCV**

[OpenCV](https://opencv.org/) is an open-source computer vision library used for image processing and working with visual data.

In GestureX, OpenCV is used to:
- Capture video from the webcam.
- Convert frames between BGR and RGB color formats.
- Draw lines between hand landmarks.
- Display the processed video.

Documentation: https://docs.opencv.org/

**MediaPipe**

[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) is Google's framework for building machine learning-based perception solutions.

I'm using its Hand Landmarker task, which detects hands and identifies 21 landmarks on each hand.

In GestureX, MediaPipe handles:
- Hand detection.
- Landmark detection.
- Hand tracking.
- Providing landmark coordinates for further processing.

Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

## How It Works

1. OpenCV captures frames from the webcam.
2. Each frame is converted from BGR to RGB.
3. MediaPipe processes the frame and detects the hands.
4. The 21 landmarks of each detected hand are returned.
5. OpenCV draws connections between the landmarks.
6. The processed frame is displayed on the screen.

## Project Structure

```text
GestureX/
│
├── main.py
├── hand_landmarker.task
├── README.md
└── LICENSE
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/GestureX.git
```

**2. Navigate to the project directory**

```bash
cd GestureX
```

**3. Install the required libraries**

```bash
pip install opencv-python mediapipe
```

**4. Download the MediaPipe Hand Landmarker model**

Download the model from the official documentation:

https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models

Place the downloaded `hand_landmarker.task` file inside the project directory.

**5. Run the project**

```bash
python main.py
```

Press `q` to close the webcam window.

## Current Features

- Real-time hand detection using a webcam.
- Supports up to two hands.
- Tracks 21 landmarks per hand.
- Draws connections between finger landmarks.
- Uses MediaPipe's LIVE_STREAM running mode.

## Future Improvements

This is still a learning project, and I plan to explore more computer vision concepts as I continue developing it.

Some things I'd like to work on:
- Hand gesture recognition.
- Static hand sign detection.
- Converting recognized hand signs into text and speech.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
