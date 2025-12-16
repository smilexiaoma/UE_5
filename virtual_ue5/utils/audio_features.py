"""
音频特征提取模块
提取MFCC、pitch、energy等音频特征用于表情驱动
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioFeatureExtractor:
    """音频特征提取器"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 80,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """提取MFCC特征"""
        if not LIBROSA_AVAILABLE:
            # 模拟MFCC特征
            n_frames = len(audio) // self.hop_length + 1
            return np.random.randn(self.n_mfcc, n_frames).astype(np.float32)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return mfcc.astype(np.float32)

    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """提取音高特征"""
        if not LIBROSA_AVAILABLE:
            n_frames = len(audio) // self.hop_length + 1
            return np.random.randn(n_frames).astype(np.float32) * 100 + 200

        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        pitch = np.max(pitches, axis=0)
        return pitch.astype(np.float32)

    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """提取能量特征"""
        if not LIBROSA_AVAILABLE:
            n_frames = len(audio) // self.hop_length + 1
            return np.abs(np.random.randn(n_frames)).astype(np.float32)

        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
        )[0]
        return rms.astype(np.float32)

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """提取Mel频谱图"""
        if not LIBROSA_AVAILABLE:
            n_frames = len(audio) // self.hop_length + 1
            return np.random.randn(self.n_mels, n_frames).astype(np.float32)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)

    def extract_all_features(
        self,
        audio: np.ndarray
    ) -> dict:
        """提取所有音频特征"""
        mfcc = self.extract_mfcc(audio)
        pitch = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        mel = self.extract_mel_spectrogram(audio)

        # 确保所有特征的时间维度一致
        min_len = min(mfcc.shape[1], len(pitch), len(energy), mel.shape[1])

        return {
            'mfcc': mfcc[:, :min_len],        # (n_mfcc, T)
            'pitch': pitch[:min_len],          # (T,)
            'energy': energy[:min_len],        # (T,)
            'mel': mel[:, :min_len],           # (n_mels, T)
        }


class AudioEncoder(nn.Module):
    """
    音频编码器 - 将音频特征编码为隐向量
    用于表情驱动模型的音频分支
    """

    def __init__(
        self,
        input_dim: int = 80,  # Mel频谱维度
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

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
            x: (batch, seq_len, input_dim) 音频特征序列
        Returns:
            (batch, seq_len, output_dim) 编码后的音频特征
        """
        x = self.input_proj(x)  # (B, T, hidden_dim)
        x, _ = self.lstm(x)     # (B, T, hidden_dim*2)
        x = self.output_proj(x) # (B, T, output_dim)
        return x


class ConvAudioEncoder(nn.Module):
    """
    卷积音频编码器 - 更适合捕获局部时间特征
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        output_dim: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 音频特征序列
        Returns:
            (batch, seq_len, output_dim) 编码后的音频特征
        """
        x = x.transpose(1, 2)  # (B, input_dim, T)
        x = self.input_proj(x)  # (B, hidden_dim, T)

        # 多尺度特征提取
        conv_outputs = [conv(x) for conv in self.conv_layers]
        x = torch.cat(conv_outputs, dim=1)  # (B, hidden_dim*n, T)

        x = self.fusion(x)      # (B, hidden_dim, T)
        x = self.output_proj(x) # (B, output_dim, T)

        return x.transpose(1, 2)  # (B, T, output_dim)
