import cv2

min_confidence = 0.50

video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video_capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)

class_names = []

with open("coco.names", "r") as f:
    class_names = f.read().splitlines()

model_config = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model_weights = "frozen_inference_graph.pb"
detection_model = cv2.dnn_DetectionModel(model_config, model_weights)


detection_model.setInputSize(320, 320)
detection_model.setInputScale(1.0 / 127.5)
detection_model.setInputMean((127.5, 127.5, 127.5))
detection_model.setInputSwapRB(True)

while True:
    success, frame = video_capture.read()

    if not success:
        break

    class_ids, confidences, bbox = detection_model.detect(frame, confThreshold=min_confidence)

    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, class_names[class_id - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 150, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
