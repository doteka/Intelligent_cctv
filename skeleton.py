import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import numpy as np

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

# 1. Mediapipe 세팅
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 2. 모델 로드
# 사용자 모델을 불러옵니다.
model = STGCN_LSTM_Model()
model.load_state_dict(torch.load('/content/drive/MyDrive/dataset_action_split/best_model.pth'))
model.eval()

# 3. 웹캠 세팅
cap = cv2.VideoCapture(0)  # 0은 PC의 기본 웹캠을 의미합니다.

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
