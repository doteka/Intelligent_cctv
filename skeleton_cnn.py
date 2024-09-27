import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

# ST-GCN 레이어 정의
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_nodes):
        super(STGCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.A = A  # Adjacency matrix (스켈레톤 관절 간 연결 관계)
        self.num_nodes = num_nodes

    def forward(self, x):
        # x: (batch_size, in_channels, num_nodes, seq_length)
        h = torch.matmul(self.A, x)  # 인접 행렬을 통해 공간적 관계를 적용
        h = self.conv(h)
        return h

# ST-GCN + LSTM 기반 행동 인식 모델 정의
class STGCN_LSTM_Model(nn.Module):
    def __init__(self, num_nodes=33, in_channels=3, hidden_gcn=64, lstm_hidden_size=128, num_classes=8, seq_length=30):
        super(STGCN_LSTM_Model, self).__init__()

        # 인접 행렬 A 정의 (관절 간의 연결 관계)
        self.A = torch.eye(num_nodes)  # 간단한 예시로 단위 행렬로 시작

        # ST-GCN 레이어
        self.stgcn_layer = STGCNLayer(in_channels, hidden_gcn, self.A, num_nodes)
        self.stgcn_layer2 = STGCNLayer(in_channels, hidden_gcn*2, self.A, num_nodes)

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=hidden_gcn * 2 * num_nodes, hidden_size=hidden_gcn, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_gcn, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, num_nodes, in_channels)

        batch_size, seq_length, num_nodes, in_channels = x.size()

        # Reshape input to fit into ST-GCN: (batch_size, in_channels, num_nodes, seq_length)
        x = x.permute(0, 3, 2, 1)

        # Apply ST-GCN
        gcn_out = self.stgcn_layer(x)
        gcn_out = self.stgcn_layer2(x)

        # Flatten the output for LSTM: (batch_size, seq_length, num_nodes * hidden_gcn)
        gcn_out = gcn_out.permute(0, 3, 2, 1).contiguous().view(batch_size, seq_length, -1)

        # Apply LSTM
        lstm_out, _ = self.lstm(gcn_out)
        lstm_out, _ = self.lstm2(lstm_out)

        # Take the last output of LSTM (for classification)
        lstm_out_last = lstm_out[:, -1, :]

        # Apply final classification layer
        out = self.fc(lstm_out_last)

        return out

# st-gcn:3 + lstm: 1
class STGCN_LSTM_Model_V2(nn.Module):
    def __init__(self, num_nodes=33, in_channels=3, hidden_gcn=64, lstm_hidden_size=128, num_classes=8, seq_length=30):
        super(STGCN_LSTM_Model_V2, self).__init__()

        self.A = torch.eye(num_nodes)

        # 3개의 ST-GCN 레이어
        self.stgcn_layer1 = STGCNLayer(in_channels, hidden_gcn, self.A, num_nodes)
        self.stgcn_layer2 = STGCNLayer(hidden_gcn, hidden_gcn * 2, self.A, num_nodes)
        self.stgcn_layer3 = STGCNLayer(hidden_gcn * 2, hidden_gcn * 4, self.A, num_nodes)

        # 1개의 LSTM 레이어
        self.lstm = nn.LSTM(input_size=hidden_gcn * 4 * num_nodes, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, num_nodes, in_channels = x.size()

        # ST-GCN 적용
        x = x.permute(0, 3, 2, 1)  # (batch_size, in_channels, num_nodes, seq_length)
        gcn_out = self.stgcn_layer1(x)
        gcn_out = self.stgcn_layer2(gcn_out)
        gcn_out = self.stgcn_layer3(gcn_out)

        # Flatten for LSTM
        gcn_out = gcn_out.permute(0, 3, 2, 1).contiguous().view(batch_size, seq_length, -1)

        # LSTM 적용
        lstm_out, _ = self.lstm(gcn_out)

        # LSTM의 마지막 출력 사용
        lstm_out_last = lstm_out[:, -1, :]

        # Fully connected layer for classification
        out = self.fc(lstm_out_last)

        return out

# st-gcn:2 + lstm: 1
class STGCN_LSTM_Model_V3(nn.Module):
    def __init__(self, num_nodes=33, in_channels=3, hidden_gcn=64, lstm_hidden_size=128, num_classes=8, seq_length=30):
        super(STGCN_LSTM_Model_V3, self).__init__()

        self.A = torch.eye(num_nodes)

        # 3개의 ST-GCN 레이어
        self.stgcn_layer1 = STGCNLayer(in_channels, hidden_gcn, self.A, num_nodes)
        self.stgcn_layer2 = STGCNLayer(hidden_gcn, hidden_gcn * 2, self.A, num_nodes)

        # 1개의 LSTM 레이어
        self.lstm = nn.LSTM(input_size=hidden_gcn * 2 * num_nodes, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, num_nodes, in_channels = x.size()

        # ST-GCN 적용
        x = x.permute(0, 3, 2, 1)  # (batch_size, in_channels, num_nodes, seq_length)
        gcn_out = self.stgcn_layer1(x)
        gcn_out = self.stgcn_layer2(gcn_out)

        # Flatten for LSTM
        gcn_out = gcn_out.permute(0, 3, 2, 1).contiguous().view(batch_size, seq_length, -1)

        # LSTM 적용
        lstm_out, _ = self.lstm(gcn_out)

        # LSTM의 마지막 출력 사용
        lstm_out_last = lstm_out[:, -1, :]

        # Fully connected layer for classification
        out = self.fc(lstm_out_last)

        return out

# 프레임 버퍼 초기화
frame_buffer = []

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 2. ONNX 모델 로드 (양자화된 모델)
onnx_model_path = './Skeleton_action_quant.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# 상태 목록
states = ['A', 'Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand Up', 'Standing', 'Walking']
seq_length = 1

# 3. 웹캠 세팅
cap = cv2.VideoCapture(1)  # 1은 외부 웹캠, 0은 PC 기본 웹캠을 의미합니다.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 현재 프레임 처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 스켈레톤 좌표 추출
        skeleton = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).flatten()
        
        # 현재 스켈레톤을 프레임 버퍼에 추가
        frame_buffer.append(skeleton)
        
        # 프레임 버퍼가 seq_length에 도달했는지 확인
        if len(frame_buffer) == seq_length:
            skeleton_input = np.array(frame_buffer).reshape(1, seq_length, 33, 3).astype(np.float32)  # (batch_size, seq_length, num_nodes, in_channels)

            # ONNX 모델로 추론 수행
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            output = ort_session.run([output_name], {input_name: skeleton_input})

            predicted_state = states[np.argmax(output)]
            # 분류된 상태를 영상에 출력
            cv2.putText(frame, f'State: {predicted_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 프레임 버퍼 초기화
            frame_buffer.pop(0)  # 첫 번째 프레임 제거
    # 영상 출력
    cv2.imshow('Action Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# # 1. Mediapipe 세팅
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# # 2. ONNX 모델 로드 (양자화된 모델)
# onnx_model_path = './Skeleton_action_quant.onnx'
# ort_session = ort.InferenceSession(onnx_model_path)

# # 상태 목록
# states = ['A', 'Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand Up', 'Standing', 'Walking']

# # 3. 웹캠 세팅
# cap = cv2.VideoCapture(1)  # 1은 외부 웹캠, 0은 PC 기본 웹캠을 의미합니다.

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # 4. Mediapipe를 사용해 스켈레톤 추출
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
    
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
#         # 스켈레톤 좌표 추출 (33개의 좌표, 각각 x, y, z)
#         skeleton = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).flatten()

#         # 5. ONNX 모델에 전달할 입력 데이터 준비
#         skeleton_input = np.expand_dims(skeleton, axis=0)  # 배치 차원 추가 (1, 99)
#         skeleton_input = np.expand_dims(skeleton_input, axis=1)  # seq_length 차원 추가 (1, 1, 99)
#         skeleton_input = skeleton_input.reshape(1, 1, 33, 3).astype(np.float32)  # (batch_size, seq_length, num_nodes, in_channels)

#         # 6. 양자화된 ONNX 모델을 사용해 추론 수행
#         input_name = ort_session.get_inputs()[0].name
#         output_name = ort_session.get_outputs()[0].name
#         output = ort_session.run([output_name], {input_name: skeleton_input})

#         predicted_state = states[np.argmax(output)]
        
#         # 7. 분류된 상태를 영상에 출력
#         cv2.putText(frame, f'State: {predicted_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     # 영상 출력
#     cv2.imshow('Action Recognition', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# 양자화 하지 않은 원본 .pht 실행
'''
# 1. Mediapipe 세팅
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 2. 모델 로드
# 사용자 모델을 불러옵니다.
model = STGCN_LSTM_Model_V3()
model.load_state_dict(torch.load('./best_model_seq_length_ST_GCN3_LSTM1_frame50.pth'))
model.eval()

# 3. 웹캠 세팅
cap = cv2.VideoCapture(1)  # 0은 PC의 기본 웹캠을 의미합니다.

# 상태 목록
states = ['A', 'Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand Up', 'Standing', 'Walking']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. Mediapipe를 사용해 스켈레톤 추출
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 스켈레톤 좌표 추출 (33개의 좌표, 각각 x, y, z)
        skeleton = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).flatten()

        # 5. 스켈레톤 데이터를 PyTorch 모델로 분류
        skeleton_tensor = torch.tensor(skeleton).float().unsqueeze(0)  # 배치 차원 추가
        skeleton_tensor = skeleton_tensor.unsqueeze(1)  # seq_length 차원 추가
        skeleton_tensor = skeleton_tensor.view(1, 1, 33, 3)  # (batch_size, seq_length, num_nodes, in_channels)

        with torch.no_grad():
            output = model(skeleton_tensor)
            predicted_state = states[torch.argmax(output).item()]
        
        # 6. 분류된 상태를 영상에 출력
        cv2.putText(frame, f'State: {predicted_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 영상 출력
    cv2.imshow('Action Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
