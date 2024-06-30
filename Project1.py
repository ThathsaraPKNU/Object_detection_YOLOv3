import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
from random import shuffle

# Initialize the parameters
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4   # Non-maximum suppression threshold
width = 416           # Width of network's input image
height = 416          # Height of network's input image

# Load names of classes
classes_file = r"D:\OpenCV_projects\coco_classes.txt"  # Update with your classes file path
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the model
model_configuration = r"D:\OpenCV_projects\yolov3.cfg"       # Update with your model configuration path
model_weights = r"D:\OpenCV_projects\yolov3.weights"          # Update with your model weights path
net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def get_output_layers(net):
    layersNames = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    if isinstance(output_layers_indices, int):
        output_layers_indices = [output_layers_indices]
    output_layers = [layersNames[i - 1] for i in output_layers_indices]
    return output_layers

# Draw bounding boxes
def draw_boxes(img, class_id, conf, left, top, right, bottom):
    label = "{}: {:.2f}%".format(classes[class_id], conf * 100)
    # Generate random colors for each class (optional)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(classes))]
    color = tuple([int(255*x) for x in colors[class_id]])
    top = top - 15 if top - 15 > 15 else top + 15

    pil_im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # Load font and calculate text width
    font_path = r"D:\OpenCV_projects\verdana.ttf"  # Update with your font path
    font = ImageFont.truetype(font_path, 25)
    label_bbox = draw.textbbox((left, top), label, font=font)
    label_width = label_bbox[2] - label_bbox[0]
    label_height = label_bbox[3] - label_bbox[1]

    # Draw rectangle for the label
    if top - label_height >= 0:
        text_origin = np.array([left, top - label_height])
    else:
        text_origin = np.array([left, top + 1])

    for i in range((img.shape[0] + img.shape[1]) // 300):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)

    # Draw filled rectangle for the text background
    draw.rectangle([tuple(text_origin), tuple(text_origin + (label_width, label_height))], fill=color)

    # Draw text on the image
    draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)

    del draw

    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return img

# Perform non-max suppression
def post_process(img, outs):
    height = img.shape[0]
    width = img.shape[1]
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                box_width = int(detection[2] * width)
                box_height = int(detection[3] * height)
                left = int(center_x - box_width / 2)
                top = int(center_y - box_height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, box_width, box_height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i.item()
        box = boxes[i]
        left = box[0]
        top = box[1]
        box_width = box[2]
        box_height = box[3]
        img = draw_boxes(img, class_ids[i], confidences[i], left, top, left + box_width, top + box_height)

    return img

# Generate random colors for each class
hsv_tuples = [(x/len(classes), 1., 1.) for x in range(len(classes))]
shuffle(hsv_tuples)
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

# Main process
cap = cv2.VideoCapture(0)  # 0 for default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    orig = frame.copy()

    blob = cv2.dnn.blobFromImage(img, 1/255, (width, height), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    img = post_process(img, outs)

    cv2.imshow('YOLO Object Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
