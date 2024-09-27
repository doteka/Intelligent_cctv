import cv2
import numpy as np
import onnxruntime
import mediapipe as mp

class CONFIG:
    onnx_model_path = './model/yolov5n_custom_dataset_train_model_quant.onnx'
    action_model_path = './model/action.onnx'  # Path to the action model
    img_size = (640, 640)
    conf_threshold = 0.35
    iou_threshold = 0.4
    camera_index = 0  # Camera index (default camera)

# Object list and their respective colors
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
colors = {
    'Hardhat': (0, 255, 0),       # Green
    'Mask': (255, 0, 0),          # Blue
    'NO-Hardhat': (0, 0, 255),    # Red
    'NO-Mask': (0, 165, 255),     # Orange
    'NO-Safety Vest': (255, 255, 0),  # Cyan
    'Person': (255, 0, 255),      # Magenta
    'Safety Cone': (0, 255, 255), # Yellow
    'Safety Vest': (128, 0, 128), # Purple
    'machinery': (128, 128, 0),   # Olive
    'vehicle': (128, 0, 0),       # Maroon
}

def preprocess_image(image, size=CONFIG.img_size):
    h, w, _ = image.shape
    scale = min(size[0] / h, size[1] / w)
    
    # Resize the image while maintaining the aspect ratio
    new_w, new_h = int(scale * w), int(scale * h)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Create a new image and pad it centrally
    padded_image = np.full((size[1], size[0], 3), 114, dtype=np.uint8)
    padded_image[:new_h, :new_w, :] = resized_image

    # Convert to RGB and normalize
    image_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_preprocessed = np.expand_dims(image_transposed, axis=0)
    return image_preprocessed, scale, new_w, new_h

def postprocess_output(output, conf_threshold=0.25):
    boxes = []
    for detection in output:
        confidence = detection[4]
        if confidence > conf_threshold:
            x_center, y_center, width, height = detection[:4]
            x1 = int((x_center - width / 2))
            y1 = int((y_center - height / 2))
            x2 = int((x_center + width / 2))
            y2 = int((y_center + height / 2))
            class_id = int(np.argmax(detection[5:]))  # Get class ID
            boxes.append([x1, y1, x2, y2, confidence, class_id])
    return boxes

def nms(boxes, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def draw_skeleton(frame, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

def classify_action(skeleton_data, ort_session):
    # Preprocess skeleton data for action model (modify this part based on your action model's input)
    skeleton_data = np.array(skeleton_data).astype(np.float32)
    skeleton_data = np.expand_dims(skeleton_data, axis=0)  # Add batch dimension
    ort_inputs = {ort_session.get_inputs()[0].name: skeleton_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Assuming the model outputs class probabilities
    action_id = np.argmax(ort_outputs[0])
    action_labels = ['Action1', 'Action2', 'Action3']  # Replace with your actual action labels
    return action_labels[action_id]

def main():
    # Load ONNX models
    detection_model_path = CONFIG.onnx_model_path
    action_model_path = CONFIG.action_model_path
    detection_session = onnxruntime.InferenceSession(detection_model_path)
    action_session = onnxruntime.InferenceSession(action_model_path)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Start camera capture
    camera_index = CONFIG.camera_index
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess image for detection
        pre_image, scale, new_w, new_h = preprocess_image(frame)
        
        # Model inference for detection
        ort_inputs = {detection_session.get_inputs()[0].name: pre_image.astype(np.float32)}
        ort_outputs = detection_session.run(None, ort_inputs)
        
        # Post-process output and apply NMS
        boxes = postprocess_output(ort_outputs[0][0], conf_threshold=CONFIG.conf_threshold)
        nms_indices = nms(boxes, iou_threshold=CONFIG.iou_threshold)
        boxes = [boxes[i] for i in nms_indices]

        # Draw bounding boxes and extract skeleton
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box
            class_name = class_names[class_id]
            color = colors[class_name]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display object name and confidence
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # If 'Person' is detected, extract skeleton and classify action
            if class_name == 'Person':
                # Ensure bounding box coordinates are within image dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Crop the bounding box area for the person
                person_roi = frame[y1:y2, x1:x2]
                
                if person_roi.size > 0:
                    # Extract skeleton
                    person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    result = pose.process(person_roi_rgb)
                    
                    # Draw skeleton
                    if result.pose_landmarks:
                        draw_skeleton(person_roi, result.pose_landmarks)
                        
                        # Prepare skeleton data for action classification
                        skeleton_data = []
                        for landmark in result.pose_landmarks.landmark:
                            skeleton_data.append([landmark.x, landmark.y, landmark.z])  # Assuming (x, y, z) format
                        
                        # Classify action
                        action_label = classify_action(skeleton_data, action_session)
                        # Update label with action
                        label = f'{class_name}: {confidence:.2f} (Action: {action_label})'
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display resulting image
        cv2.imshow("Detected Objects", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
