"""
视频处理工具 - 使用 MediaPipe 提取面部关键点
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import torch


class VideoProcessor:
    """视频处理器 - 提取面部关键点"""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        初始化 MediaPipe Face Mesh

        Args:
            max_num_faces: 最大检测人脸数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_landmarks_from_frame(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        从单帧图像中提取面部关键点

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            landmarks: (478, 3) 关键点坐标,如果未检测到则返回 None
        """
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测面部关键点
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # 提取第一个人脸的关键点
        face_landmarks = results.multi_face_landmarks[0]

        # 转换为 numpy 数组 (478, 3)
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in face_landmarks.landmark
        ], dtype=np.float32)

        return landmarks

    def process_video(
        self,
        video_path: str,
        target_fps: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        处理视频文件,提取所有帧的面部关键点

        Args:
            video_path: 视频文件路径
            target_fps: 目标帧率,如果为 None 则使用原始帧率

        Returns:
            landmarks_seq: (T, 478, 3) 关键点序列
            fps: 实际帧率
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算采样间隔
        if target_fps is None or target_fps >= original_fps:
            frame_interval = 1
            fps = original_fps
        else:
            frame_interval = int(original_fps / target_fps)
            fps = original_fps / frame_interval

        landmarks_list = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔采样
            if frame_idx % frame_interval == 0:
                landmarks = self.extract_landmarks_from_frame(frame)

                if landmarks is not None:
                    landmarks_list.append(landmarks)
                else:
                    # 如果检测失败,使用零填充或前一帧
                    if len(landmarks_list) > 0:
                        landmarks_list.append(landmarks_list[-1])
                    else:
                        landmarks_list.append(np.zeros((478, 3), dtype=np.float32))

            frame_idx += 1

        cap.release()

        if len(landmarks_list) == 0:
            raise ValueError(f"未能从视频中提取任何关键点: {video_path}")

        # 转换为 numpy 数组
        landmarks_seq = np.stack(landmarks_list, axis=0)  # (T, 478, 3)

        return landmarks_seq, fps

    def landmarks_to_features(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        将关键点转换为特征向量

        Args:
            landmarks: (T, 478, 3) 或 (478, 3)

        Returns:
            features: (T, 1434) 或 (1434,) 展平的特征
        """
        original_shape = landmarks.shape

        if len(original_shape) == 2:
            # 单帧: (478, 3) -> (1434,)
            return landmarks.reshape(-1)
        else:
            # 序列: (T, 478, 3) -> (T, 1434)
            return landmarks.reshape(landmarks.shape[0], -1)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class BlendShapeEstimator:
    """
    从面部关键点估计 52 维 BlendShape 权重
    使用简化的线性映射方法
    """

    def __init__(self):
        """初始化 BlendShape 估计器"""
        # 定义关键区域的关键点索引
        self.mouth_indices = list(range(61, 81)) + list(range(291, 311))  # 嘴部
        self.left_eye_indices = list(range(33, 42)) + list(range(133, 144))  # 左眼
        self.right_eye_indices = list(range(362, 371)) + list(range(263, 274))  # 右眼
        self.left_eyebrow_indices = list(range(70, 80))  # 左眉
        self.right_eyebrow_indices = list(range(300, 310))  # 右眉
        self.jaw_indices = list(range(0, 17))  # 下颌

    def estimate_blendshapes(
        self,
        landmarks: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        从关键点估计 BlendShape 权重

        Args:
            landmarks: (T, 478, 3) 或 (478, 3) 关键点坐标
            normalize: 是否归一化到 [0, 1]

        Returns:
            blendshapes: (T, 52) 或 (52,) BlendShape 权重
        """
        is_sequence = len(landmarks.shape) == 3

        if not is_sequence:
            landmarks = landmarks[np.newaxis, ...]  # (1, 478, 3)

        T = landmarks.shape[0]
        blendshapes = np.zeros((T, 52), dtype=np.float32)

        for t in range(T):
            lm = landmarks[t]  # (478, 3)

            # 计算各区域的运动特征
            # 1-8: 嘴部 BlendShapes
            mouth_lm = lm[self.mouth_indices]
            mouth_height = np.max(mouth_lm[:, 1]) - np.min(mouth_lm[:, 1])
            mouth_width = np.max(mouth_lm[:, 0]) - np.min(mouth_lm[:, 0])
            blendshapes[t, 0] = mouth_height * 10  # jawOpen
            blendshapes[t, 1] = mouth_width * 5    # mouthSmile

            # 9-16: 眼部 BlendShapes
            left_eye_lm = lm[self.left_eye_indices]
            right_eye_lm = lm[self.right_eye_indices]
            left_eye_height = np.max(left_eye_lm[:, 1]) - np.min(left_eye_lm[:, 1])
            right_eye_height = np.max(right_eye_lm[:, 1]) - np.min(right_eye_lm[:, 1])
            blendshapes[t, 8] = 1.0 - left_eye_height * 20   # eyeBlinkLeft
            blendshapes[t, 9] = 1.0 - right_eye_height * 20  # eyeBlinkRight

            # 17-24: 眉毛 BlendShapes
            left_brow_lm = lm[self.left_eyebrow_indices]
            right_brow_lm = lm[self.right_eyebrow_indices]
            left_brow_y = np.mean(left_brow_lm[:, 1])
            right_brow_y = np.mean(right_brow_lm[:, 1])
            blendshapes[t, 16] = (0.3 - left_brow_y) * 3   # browInnerUp
            blendshapes[t, 17] = (0.3 - right_brow_y) * 3

            # 25-52: 其他面部特征 (简化处理)
            # 使用关键点的统计特征
            jaw_lm = lm[self.jaw_indices]
            jaw_width = np.max(jaw_lm[:, 0]) - np.min(jaw_lm[:, 0])
            blendshapes[t, 24] = jaw_width * 2  # cheekPuff

            # 填充剩余的 BlendShapes (使用关键点的方差作为特征)
            for i in range(25, 52):
                region_idx = i % len(lm)
                blendshapes[t, i] = np.std(lm[region_idx]) * 5

        # 归一化到 [0, 1]
        if normalize:
            blendshapes = np.clip(blendshapes, 0, 1)

        if not is_sequence:
            blendshapes = blendshapes[0]  # (52,)

        return blendshapes

    def estimate_head_pose(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        从关键点估计头部姿态 (四元数)

        Args:
            landmarks: (T, 478, 3) 或 (478, 3)

        Returns:
            quaternions: (T, 4) 或 (4,) 四元数 [w, x, y, z]
        """
        is_sequence = len(landmarks.shape) == 3

        if not is_sequence:
            landmarks = landmarks[np.newaxis, ...]

        T = landmarks.shape[0]
        quaternions = np.zeros((T, 4), dtype=np.float32)

        for t in range(T):
            lm = landmarks[t]

            # 使用鼻尖、左右眼角等关键点估计头部方向
            nose_tip = lm[1]  # 鼻尖
            left_eye = lm[33]  # 左眼角
            right_eye = lm[263]  # 右眼角

            # 计算头部朝向向量
            eye_center = (left_eye + right_eye) / 2
            forward = nose_tip - eye_center

            # 简化的四元数计算 (假设小角度旋转)
            pitch = forward[1] * 2  # 俯仰
            yaw = forward[0] * 2    # 偏航
            roll = (left_eye[1] - right_eye[1]) * 2  # 翻滚

            # 转换为四元数 (简化版本)
            quaternions[t, 0] = 1.0  # w
            quaternions[t, 1] = pitch  # x
            quaternions[t, 2] = yaw    # y
            quaternions[t, 3] = roll   # z

            # 归一化
            norm = np.linalg.norm(quaternions[t])
            if norm > 0:
                quaternions[t] /= norm

        if not is_sequence:
            quaternions = quaternions[0]

        return quaternions
