import cv2 as cv  # Import OpenCV for computer vision
import mediapipe as mp  # Import MediaPipe for hand gesture detection
import time as t


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.tasks.vision.drawing_utils


# Create a list to store the processed images
image_ = []


# Callback function for processing hand landmark detection results
def print_result(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    # Define the landmark indices for each part of the hand
    WRIST = 0

    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4

    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8

    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12

    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16

    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # Group landmarks according to their respective fingers
    FINGER_LANDMARK_GROUPS = [
        [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
        [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
        [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
        [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
        [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]
    ]

    # Create a copy of the output image for drawing
    draw_image = output_image.numpy_view().copy()

    # Loop through the detected hands
    for i in range(0, 2):
        if result.hand_landmarks:
            hand = result.hand_landmarks[i]

            height, width = draw_image.shape[:2]

            # Draw connections between the landmarks of each finger
            for group in FINGER_LANDMARK_GROUPS:
                for i in range(len(group) - 1):
                    start = hand[group[i]]
                    end = hand[group[i + 1]]

                    end_point = (
                        int(start.x * width),
                        int(start.y * height)
                    )

                    start_point = (
                        int(end.x * width),
                        int(end.y * height)
                    )

                    cv.line(draw_image, start_point, end_point, (0, 255, 0), 3)

                    image_.append(draw_image)


# Configure the Hand Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=r'hand_landmarker.task'
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2
)


# Start the webcam and run hand landmark detection
with HandLandmarker.create_from_options(options) as landmarker:
    capture = cv.VideoCapture(0)

    while True:
        isTrue, frame = capture.read()

        # Convert the webcam frame from BGR to RGB
        SRGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the frame into a MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=SRGB_frame
        )

        image = 0

        # Display the latest processed image
        if image_:
            image = image_[-1]
            BGR_Image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            cv.imshow("Webcam", BGR_Image)

        # Perform asynchronous hand landmark detection
        landmarker.detect_async(mp_image, int(t.time() * 1000))

        # Press 'q' to exit the webcam
        if cv.waitKey(20) & 0xFF == ord('q'):
            break


# Release the webcam and close all OpenCV windows
capture.release()
cv.destroyAllWindows()