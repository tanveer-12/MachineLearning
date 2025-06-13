import pip
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

labels = [
    "???",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "???",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "???",
    "???",
]

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to capture frame")
        break

    h, w, _ = frame.shape
    img = cv2.resize(frame, (300, 300))
    rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    rgb_tensor = tf.convert_to_tensor(
        rgb, dtype=tf.uint8
    )  # converting the numpy array into tensorflow for tensorflow
    rgb_tensor = tf.expand_dims(
        rgb_tensor, 0
    )  # adding extra dimension in it for batch sizing
    outputs = model(rgb_tensor)
    outputs = {k: v.numpy() for k, v in outputs.items()}

    for i in range(len(outputs["detection_scores"])):
        score = outputs["detection_scores"][i]
        if tf.reduce_any(score < 0.3):
            continue

        box = outputs["detection_boxes"][i]
        class_id = int(outputs["detection_classes"][i])
        label = labels[class_id] if class_id < len(labels) else "unknown"

        y1, x1, y2, x2 = box
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, yy2 = int(y1 * h), int(y2, h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
