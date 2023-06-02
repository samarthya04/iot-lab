import cv2 as cv
import mediapipe as mp
from scipy.spatial import distance as dis
import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='MAR')
line2, = ax.plot([], [], label='EAR')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.legend()


def update_plot(x, y1, y2):
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    fig.canvas.draw()
    fig.savefig('plot.png')


def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]

    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]

        point_scale = ((int)(point.x * width), (int)(point.y * height))

        cv.circle(image, point_scale, 2, color, 1)


def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]

    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)

    distance = dis.euclidean(point1, point2)
    return distance


def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]

    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]

    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]

    left_right_dis = euclidean_distance(image, left, right)

    aspect_ratio = top_bottom_dis / left_right_dis

    return aspect_ratio


face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0, 0, 255), thickness=1, circle_radius=1)

STATIC_IMAGE = False
MAX_NO_FACES = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces=MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

capture = cv.VideoCapture(0)

frame_count = 0
min_frame = 6
min_tolerance = 0.2

total = 0
drowsy = 0
y_1 = np.zeros(100)
y_2 = np.zeros(100)
blink_counter = 0

while True:
    result, image = capture.read()
    x_plot = np.arange(100)

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)
        total = total + 1
        if outputs.multi_face_landmarks:

            draw_landmarks(image, outputs, FACE, COLOR_GREEN)

            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_RED)

            ratio_left = get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)

            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)

            ratio_right = get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)

            ratio = (ratio_left + ratio_right) / 2.0
            y_1[total % 100] = ratio
            if ratio < min_tolerance:
                frame_count += 1
            else:
                frame_count = 0

            if frame_count > min_frame:
                blink_counter += 1

            draw_landmarks(image, outputs, UPPER_LOWER_LIPS, COLOR_BLUE)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)

            ratio_lips = get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
            y_2[total % 100] = ratio_lips
            if ratio_lips > 0.60:
                drowsy = drowsy + 1

        cv.putText(image, "Blink Count: %d" % (blink_counter), (int(0.02 * image.shape[1]), int(0.6 * image.shape[0])),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        cv.putText(image, "EAR: %.2f" % (ratio), (int(0.02 * image.shape[1]), int(0.7 * image.shape[0])),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        cv.putText(image, "MAR: %.2f" % (ratio_lips), (int(0.02 * image.shape[1]), int(0.8 * image.shape[0])),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        cv.putText(image, "PERCLOS: %.2f" % (drowsy / total), (int(0.02 * image.shape[1]), int(0.9 * image.shape[0])),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        update_plot(x_plot, y_1, y_2)
        cv.imshow("FACE MESH", image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv.destroyAllWindows()