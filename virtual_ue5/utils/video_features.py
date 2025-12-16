"""
视频/面部特征提取模块
提取面部关键点、表情参数等视觉特征
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceLandmarkExtractor:
    """
    面部关键点提取器
    使用MediaPipe Face Mesh提取468个面部关键点
    """

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_mesh = None

    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        从图像中提取面部关键点

        Args:
            image: BGR图像 (H, W, 3)
        Returns:
            landmarks: (478, 3) 归一化的关键点坐标，失败返回None
        """
        if self.face_mesh is None:
            # 模拟关键点数据
            return np.random.randn(478, 3).astype(np.float32) * 0.1 + 0.5

        if CV2_AVAILABLE:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            return coords.astype(np.float32)

        return None

    def extract_landmarks_sequence(
        self,
        frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        从视频帧序列中提取关键点序列

        Args:
            frames: List of BGR images
        Returns:
            landmarks: (T, 478, 3) 关键点序列
        """
        landmarks_list = []
        prev_landmarks = None

        for frame in frames:
            landmarks = self.extract_landmarks(frame)
            if landmarks is None:
                # 使用前一帧的关键点或零填充
                if prev_landmarks is not None:
                    landmarks = prev_landmarks.copy()
                else:
                    landmarks = np.zeros((478, 3), dtype=np.float32)
            prev_landmarks = landmarks
            landmarks_list.append(landmarks)

        return np.stack(landmarks_list, axis=0)

    def close(self):
        if self.face_mesh is not None and hasattr(self.face_mesh, 'close'):
            self.face_mesh.close()


class BlendShapeExtractor:
    """
    ARKit 52 BlendShape 提取器
    将关键点转换为52维BlendShape权重
    """

    # ARKit 52 BlendShape 名称
    BLENDSHAPE_NAMES = [
        'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft',
        'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight',
        'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
        'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight',
        'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
        'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft',
        'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft',
        'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
        'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft',
        'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft',
        'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
        'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft',
        'noseSneerRight', 'tongueOut',
    ]

    def __init__(self):
        # 关键点索引映射（简化版本）
        self.lip_upper = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lip_lower = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        self.right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263]
        self.left_brow = [70, 63, 105, 66, 107]
        self.right_brow = [336, 296, 334, 293, 300]

    def landmarks_to_blendshapes(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        将面部关键点转换为BlendShape权重

        Args:
            landmarks: (478, 3) 或 (T, 478, 3) 关键点
        Returns:
            blendshapes: (52,) 或 (T, 52) BlendShape权重
        """
        if landmarks.ndim == 2:
            return self._single_frame_blendshapes(landmarks)
        else:
            return np.stack([
                self._single_frame_blendshapes(lm) for lm in landmarks
            ], axis=0)

    def _single_frame_blendshapes(self, landmarks: np.ndarray) -> np.ndarray:
        """单帧关键点到BlendShape的转换"""
        blendshapes = np.zeros(52, dtype=np.float32)

        # 眼睛开合
        left_eye_height = self._get_eye_aspect_ratio(landmarks, self.left_eye)
        right_eye_height = self._get_eye_aspect_ratio(landmarks, self.right_eye)
        blendshapes[0] = 1.0 - left_eye_height   # eyeBlinkLeft
        blendshapes[7] = 1.0 - right_eye_height  # eyeBlinkRight

        # 嘴巴开合
        lip_distance = self._get_lip_distance(landmarks)
        blendshapes[17] = np.clip(lip_distance * 3, 0, 1)  # jawOpen

        # 微笑
        smile_left, smile_right = self._get_smile_ratio(landmarks)
        blendshapes[23] = smile_left   # mouthSmileLeft
        blendshapes[24] = smile_right  # mouthSmileRight

        # 眉毛
        brow_left = self._get_brow_height(landmarks, self.left_brow)
        brow_right = self._get_brow_height(landmarks, self.right_brow)
        blendshapes[43] = np.clip(brow_left * 2, 0, 1)   # browInnerUp
        blendshapes[44] = np.clip(brow_left * 2, 0, 1)   # browOuterUpLeft
        blendshapes[45] = np.clip(brow_right * 2, 0, 1)  # browOuterUpRight

        return blendshapes

    def _get_eye_aspect_ratio(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """计算眼睛纵横比"""
        pts = landmarks[indices]
        height = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
        width = np.linalg.norm(pts[0] - pts[3])
        return float(np.clip(height / (2.0 * width + 1e-6), 0, 1))

    def _get_lip_distance(self, landmarks: np.ndarray) -> float:
        """计算上下嘴唇距离"""
        upper = landmarks[self.lip_upper].mean(axis=0)
        lower = landmarks[self.lip_lower].mean(axis=0)
        return float(np.linalg.norm(upper - lower))

    def _get_smile_ratio(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """计算微笑程度"""
        # 嘴角位置
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = (landmarks[0] + landmarks[17]) / 2

        # 嘴角相对于嘴巴中心的高度差
        left_smile = float(np.clip((mouth_center[1] - left_corner[1]) * 10, 0, 1))
        right_smile = float(np.clip((mouth_center[1] - right_corner[1]) * 10, 0, 1))

        return left_smile, right_smile

    def _get_brow_height(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """计算眉毛高度"""
        brow_pts = landmarks[indices]
        nose_tip = landmarks[4]
        height = brow_pts[:, 1].mean() - nose_tip[1]
        return float(np.clip(-height * 5, 0, 1))


class VideoEncoder(nn.Module):
    """
    视频/视觉编码器 - 将面部关键点编码为隐向量
    """

    def __init__(
        self,
        input_dim: int = 478 * 3,  # 478关键点 * 3坐标
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 关键点序列
        Returns:
            (batch, seq_len, output_dim) 编码后的视觉特征
        """
        # 展平关键点坐标
        B, T, *_ = x.shape
        x = x.view(B, T, -1)

        x = self.input_proj(x)   # (B, T, hidden_dim)
        x, _ = self.lstm(x)      # (B, T, hidden_dim*2)
        x = self.output_proj(x)  # (B, T, output_dim)

        return x


class TemporalConvVideoEncoder(nn.Module):
    """
    时序卷积视频编码器 - 更好地捕获局部时间模式
    """

    def __init__(
        self,
        input_dim: int = 478 * 3,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                          padding=padding, dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.conv_layers = nn.ModuleList(layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 关键点序列
        Returns:
            (batch, seq_len, output_dim) 编码后的视觉特征
        """
        B, T, *_ = x.shape
        x = x.view(B, T, -1)

        x = self.input_proj(x)  # (B, T, hidden_dim)
        x = x.transpose(1, 2)   # (B, hidden_dim, T)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            x = x + residual  # 残差连接

        x = x.transpose(1, 2)   # (B, T, hidden_dim)
        x = self.output_proj(x) # (B, T, output_dim)

        return x
