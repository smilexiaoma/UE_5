"""
RAVDESS 数据集加载器
"""

import os
import glob
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from .video_processor import VideoProcessor, BlendShapeEstimator
from .audio_processor import AudioProcessor


class RAVDESSDataset(Dataset):
    """
    RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) 数据集

    文件命名格式: modality-vocal_channel-emotion-intensity-statement-repetition-actor.mp4
    例如: 01-02-03-01-01-01-01.mp4
    - modality: 01=视频
    - vocal_channel: 02=语音
    - emotion: 01=中性, 02=平静, 03=快乐, 04=悲伤, 05=愤怒, 06=恐惧, 07=���恶, 08=惊讶
    - intensity: 01=正常, 02=强烈
    - statement: 01-02
    - repetition: 01-02
    - actor: 01-24
    """

    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised',
    }

    def __init__(
        self,
        data_dir: str,
        actors: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        seq_len: Optional[int] = 100,
        audio_dim: int = 80,
        video_dim: int = 1434,
        blendshape_dim: int = 52,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        target_fps: int = 30,
    ):
        """
        初始化 RAVDESS 数据集

        Args:
            data_dir: 数据集根目录 (例如: virtual_ue5/data/1188976)
            actors: 要加载的 actor 列表 (例如: ['Actor_01', 'Actor_02'])
            emotions: 要加载的情绪列表 (例如: ['happy', 'sad'])
            max_samples: 最大样本数
            seq_len: 序列长度 (如果为 None 则使用原始长度)
            audio_dim: 音频特征维度
            video_dim: 视频特征维度
            blendshape_dim: BlendShape 维度
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
            target_fps: 目标帧率
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.blendshape_dim = blendshape_dim
        self.target_fps = target_fps

        # 初始化处理器
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.blendshape_estimator = BlendShapeEstimator()

        # 设置缓存
        if cache_dir is None:
            cache_dir = os.path.join(data_dir, '.cache')
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        os.makedirs(cache_dir, exist_ok=True)

        # 扫描数据集
        self.video_files = self._scan_dataset(actors, emotions, max_samples)

        print(f"找到 {len(self.video_files)} 个视频文件")

        # 预处理数据
        self.data = []
        self._preprocess_dataset()

    def _scan_dataset(
        self,
        actors: Optional[List[str]],
        emotions: Optional[List[str]],
        max_samples: Optional[int],
    ) -> List[str]:
        """扫描数据集,获取视频文件列表"""
        video_files = []

        # 如果未指定 actors,扫描所有 Actor 目录
        if actors is None:
            actor_dirs = glob.glob(os.path.join(self.data_dir, 'Actor_*'))
            # 过滤掉带空格的目录 (如 "Actor_01 2")
            actor_dirs = [d for d in actor_dirs if ' ' not in os.path.basename(d)]
        else:
            actor_dirs = [os.path.join(self.data_dir, actor) for actor in actors]

        # 扫描每个 actor 目录
        for actor_dir in sorted(actor_dirs):
            if not os.path.isdir(actor_dir):
                continue

            # 获取该 actor 的所有视频
            videos = glob.glob(os.path.join(actor_dir, '*.mp4'))

            for video_path in sorted(videos):
                # 解析文件名
                filename = os.path.basename(video_path)
                parts = filename.replace('.mp4', '').split('-')

                if len(parts) < 7:
                    continue

                emotion_code = parts[2]
                emotion_name = self.EMOTIONS.get(emotion_code, 'unknown')

                # 过滤情绪
                if emotions is not None and emotion_name not in emotions:
                    continue

                video_files.append(video_path)

                # 限制样本数
                if max_samples is not None and len(video_files) >= max_samples:
                    return video_files

        return video_files

    def _get_cache_path(self, video_path: str) -> str:
        """获取缓存文件路径"""
        filename = os.path.basename(video_path).replace('.mp4', '.pkl')
        return os.path.join(self.cache_dir, filename)

    def _load_from_cache(self, video_path: str) -> Optional[Dict]:
        """从缓存加载数据"""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(video_path)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None

        return None

    def _save_to_cache(self, video_path: str, data: Dict):
        """保存数据到缓存"""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(video_path)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

    def _process_video(self, video_path: str) -> Optional[Dict]:
        """处理单个视频文件"""
        try:
            # 尝试从缓存加载
            cached_data = self._load_from_cache(video_path)
            if cached_data is not None:
                return cached_data

            # 提取视频关键点
            landmarks, video_fps = self.video_processor.process_video(
                video_path,
                target_fps=self.target_fps,
            )

            # 转换为特征向量
            video_features = self.video_processor.landmarks_to_features(landmarks)

            # 估计 BlendShape
            blendshapes = self.blendshape_estimator.estimate_blendshapes(landmarks)

            # 估计头部姿态
            head_pose = self.blendshape_estimator.estimate_head_pose(landmarks)

            # 提取音频特征
            audio_features, audio_fps = self.audio_processor.process_video_audio(
                video_path,
                feature_dim=self.audio_dim,
            )

            # 对齐音频和视频
            audio_features = self.audio_processor.align_features_to_video(
                audio_features,
                len(video_features),
            )

            # 解析情绪标签
            filename = os.path.basename(video_path)
            parts = filename.replace('.mp4', '').split('-')
            emotion_code = parts[2]
            emotion_name = self.EMOTIONS.get(emotion_code, 'unknown')
            emotion_id = int(emotion_code) - 1  # 0-7

            data = {
                'audio': audio_features.astype(np.float32),  # (T, audio_dim)
                'video': video_features.astype(np.float32),  # (T, video_dim)
                'blendshapes': blendshapes.astype(np.float32),  # (T, 52)
                'head_pose': head_pose.astype(np.float32),  # (T, 4)
                'emotion': emotion_id,
                'emotion_name': emotion_name,
                'video_path': video_path,
            }

            # 保存到缓存
            self._save_to_cache(video_path, data)

            return data

        except Exception as e:
            print(f"处理视频失败: {video_path}, 错误: {e}")
            return None

    def _preprocess_dataset(self):
        """预处理整个数据集"""
        print("预处理数据集...")

        for video_path in tqdm(self.video_files):
            data = self._process_video(video_path)

            if data is not None:
                self.data.append(data)

        print(f"成功加载 {len(self.data)} 个样本")

    def _pad_or_crop(self, sequence: np.ndarray, target_len: int) -> np.ndarray:
        """填充或裁剪序列到目标长度"""
        T = sequence.shape[0]

        if T == target_len:
            return sequence
        elif T > target_len:
            # 裁剪 (从中间开始)
            start = (T - target_len) // 2
            return sequence[start:start + target_len]
        else:
            # 填充 (重复最后一帧)
            pad_len = target_len - T
            last_frame = sequence[-1:].repeat(pad_len, axis=0)
            return np.concatenate([sequence, last_frame], axis=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        data = self.data[idx]

        # 获取序列
        audio = data['audio']  # (T, audio_dim)
        video = data['video']  # (T, video_dim)
        blendshapes = data['blendshapes']  # (T, 52)
        head_pose = data['head_pose']  # (T, 4)

        # 填充或裁剪到目标长度
        if self.seq_len is not None:
            audio = self._pad_or_crop(audio, self.seq_len)
            video = self._pad_or_crop(video, self.seq_len)
            blendshapes = self._pad_or_crop(blendshapes, self.seq_len)
            head_pose = self._pad_or_crop(head_pose, self.seq_len)

        # 转换为 Tensor
        return {
            'audio': torch.from_numpy(audio),
            'video': torch.from_numpy(video),
            'blendshapes': torch.from_numpy(blendshapes),
            'head_pose': torch.from_numpy(head_pose),
            'emotion': torch.tensor(data['emotion'], dtype=torch.long),
            'emotion_name': data['emotion_name'],
            'video_path': data['video_path'],
        }

    def get_emotion_distribution(self) -> Dict[str, int]:
        """获取情绪分布"""
        distribution = {}
        for data in self.data:
            emotion_name = data['emotion_name']
            distribution[emotion_name] = distribution.get(emotion_name, 0) + 1
        return distribution
