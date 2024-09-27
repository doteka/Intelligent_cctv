import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
import torch
import torch.nn as nn

class CONFIG:
    # YOLOv5 모델 경로
    object_onnx_model_path = './model/yolov5n_custom_dataset_train_model_quant.onnx'
    # 스켈레톤 액션 모델 경로
    action_onnx_model_path = './model/Skeleton_action_cnn_quant.onnx'
    img_size = (640, 640)  # 이미지 크기
    conf_threshold = 0.35  # 신뢰도 임계값
    iou_threshold = 0.4     # NMS의 IoU 임계값
    camera_index = 0        # 카메라 인덱스 (기본 카메라)

# 객체 클래스 이름 정의
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


# 이미지 전처리 함수
def preprocess_image(image, size=CONFIG.img_size):
    h, w, _ = image.shape
    scale = min(size[0] / h, size[1] / w)  # 스케일 계산
    new_w, new_h = int(scale * w), int(scale * h)
    padded_image = np.full((size[1], size[0], 3), 114, dtype=np.uint8)  # 패딩 이미지 생성
    padded_image[:new_h, :new_w, :] = cv2.resize(image, (new_w, new_h))  # 리사이즈
    image_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)  # RGB로 변환
    image_normalized = image_rgb / 255.0  # 정규화
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # 차원 전환
    image_preprocessed = np.expand_dims(image_transposed, axis=0)  # 배치 차원 추가
    return image_preprocessed

# 출력 후처리 함수
def postprocess_output(output, conf_threshold=0.25):
    boxes = []
    for detection in output:
        confidence = detection[4]
        if confidence > conf_threshold:  # 신뢰도 필터링
            x_center, y_center, width, height = detection[:4]
            x1 = int((x_center - width / 2))  # 왼쪽 상단 x좌표
            y1 = int((y_center - height / 2))  # 왼쪽 상단 y좌표
            x2 = int((x_center + width / 2))   # 오른쪽 하단 x좌표
            y2 = int((y_center + height / 2))  # 오른쪽 하단 y좌표
            class_id = int(np.argmax(detection[5:]))  # 클래스 ID
            boxes.append([x1, y1, x2, y2, confidence, class_id])  # 박스 정보 저장
    return boxes

# NMS (비최대 억제) 함수
def nms(boxes, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 영역 계산
    order = scores.argsort()[::-1]  # 점수 정렬
    keep = []  # 유지할 박스 인덱스 저장
    
    while order.size > 0:
        i = order[0]  # 가장 높은 점수의 인덱스
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)  # 넓이
        h = np.maximum(0, yy2 - yy1 + 1)  # 높이
        inter = w * h  # 교차 영역
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # IoU 계산
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # 다음 박스 인덱스
    
    return keep

# 스켈레톤 그리기 함수
def draw_skeleton(frame, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

def main():
    # ONNX 모델 로드
    ort_object_session = onnxruntime.InferenceSession(CONFIG.object_onnx_model_path)
    ort_action_session = onnxruntime.InferenceSession(CONFIG.action_onnx_model_path)  # 스켈레톤 액션 모델 로드
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(CONFIG.camera_index)  # 카메라 캡처

    states = ['A', 'Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand Up', 'Standing', 'Walking']

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 카메라에서 더 이상 읽을 수 없는 경우 종료

        pre_image = preprocess_image(frame)  # 이미지 전처리
        ort_object_inputs = {ort_object_session.get_inputs()[0].name: pre_image.astype(np.float32)}
        ort_object_outputs = ort_object_session.run(None, ort_object_inputs)  # YOLOv5 모델 예측

        boxes = postprocess_output(ort_object_outputs[0][0], conf_threshold=CONFIG.conf_threshold)  # 출력 후처리
        nms_indices = nms(boxes, iou_threshold=CONFIG.iou_threshold)  # NMS 적용
        boxes = [boxes[i] for i in nms_indices]  # NMS 후 박스 업데이트

        # 박스 그리기 및 스켈레톤 예측
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box
            
            # 객체 클래스 이름과 신뢰도 표시
            class_name = class_names[class_id]
            label = f"{class_name}: {confidence:.2f}"  # 클래스 이름과 신뢰도 포맷
            
            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_name], 2)  # 박스 그리기
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_name], 2)  # 라벨 텍스트 그리기

            #if class_name == 'Person':  # 'Person' 클래스에 대해서만 스켈레톤 예측
            # 스켈레톤 좌표 추출
            with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
                results = pose.process(frame)
                if results.pose_landmarks:  # 스켈레톤이 감지된 경우
                    draw_skeleton(frame, results.pose_landmarks)  # 스켈레톤 그리기
                    # 스켈레톤 좌표를 모델 입력 형태로 변환
                    skeleton_coords = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
                    skeleton_input = torch.tensor(skeleton_coords).float().unsqueeze(0)  # 배치 차원 추가
                    skeleton_input = skeleton_input.unsqueeze(1)  # [1, 33, 3] -> [1, 1, 33, 3]
                    action_output = ort_action_session.run(None, {ort_action_session.get_inputs()[0].name: skeleton_input.numpy().astype(np.float32)})  # 스켈레톤 모델 예측
                    action_class = np.argmax(action_output[0])  # 가장 높은 확률의 클래스
                    action_text = states[action_class]  # 액션 텍스트

                    # 액션 텍스트 그리기
                    cv2.putText(frame, action_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                

        cv2.imshow('Real-Time Action Recognition', frame)  # 프레임 표시
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

    cap.release()  # 카메라 해제
    cv2.destroyAllWindows()  # 모든 윈도우 닫기

if __name__ == "__main__":
    main()  # 메인 함수 호출
