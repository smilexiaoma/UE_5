"""
音频处理工具 - 提取 MFCC、Mel频谱等特征
"""

import numpy as np
import librosa
import subprocess
import tempfile
import os
from typing import Tuple, Optional


class AudioProcessor:
    """音频处理器 - 提取音频特征"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 80,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ):
        """
        初始化音频处理器

        Args:
            sample_rate: 采样率
            n_mfcc: MFCC 系数数量
            n_mels: Mel 频带数量
            n_fft: FFT 窗口大小
            hop_length: 帧移
            win_length: 窗口长度
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def extract_audio_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        从视频文件中提取音频

        Args:
            video_path: 视频文件路径
            output_path: 输出音频��径,如果为 None 则使用临时文件

        Returns:
            audio_path: 提取的音频文件路径
        """
        if output_path is None:
            # 创建临时文件
            temp_fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)

        # 使用 ffmpeg 提取音频
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # 不处理视频
            '-acodec', 'pcm_s16le',  # 音频编码
            '-ar', str(self.sample_rate),  # 采样率
            '-ac', '1',  # 单声道
            '-y',  # 覆盖输出文件
            output_path
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音频提取失败: {video_path}") from e

        return output_path

    def load_audio(
        self,
        audio_path: str,
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            audio: 音频波形 (T,)
            sr: 采样率
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio, sr

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        提取 Mel 频谱

        Args:
            audio: 音频波形 (T,)

        Returns:
            mel_spec: Mel 频谱 (n_mels, T_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
        )

        # 转换为 dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    def extract_mfcc(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        提取 MFCC 特征

        Args:
            audio: 音频波形 (T,)

        Returns:
            mfcc: MFCC 特征 (n_mfcc, T_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mfcc

    def extract_prosody_features(
        self,
        audio: np.ndarray,
    ) -> dict:
        """
        提取韵律特征 (pitch, energy 等)

        Args:
            audio: 音频波形 (T,)

        Returns:
            features: 包含各种韵律特征的字典
        """
        features = {}

        # 提取基频 (pitch)
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        pitch = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch.append(pitches[index, t])
        features['pitch'] = np.array(pitch)  # (T_frames,)

        # 提取能量
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )[0]
        features['energy'] = energy  # (T_frames,)

        # 提取过零率
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )[0]
        features['zcr'] = zcr  # (T_frames,)

        return features

    def extract_all_features(
        self,
        audio: np.ndarray,
        feature_dim: int = 80,
    ) -> np.ndarray:
        """
        提取所有音频特征并组合

        Args:
            audio: 音频波形 (T,)
            feature_dim: 目标特征维度

        Returns:
            features: (T_frames, feature_dim) 音频特征
        """
        # 提取 Mel 频谱
        mel_spec = self.extract_mel_spectrogram(audio)  # (n_mels, T_frames)

        # 提取 MFCC
        mfcc = self.extract_mfcc(audio)  # (n_mfcc, T_frames)

        # 提取韵律特征
        prosody = self.extract_prosody_features(audio)
        pitch = prosody['pitch'][:, np.newaxis]  # (T_frames, 1)
        energy = prosody['energy'][:, np.newaxis]  # (T_frames, 1)
        zcr = prosody['zcr'][:, np.newaxis]  # (T_frames, 1)

        # 组合特征
        # 使用 Mel 频谱作为主要特征
        if feature_dim == self.n_mels:
            # 只使用 Mel 频谱
            features = mel_spec.T  # (T_frames, n_mels)
        else:
            # 组合多种特征
            # Mel (80) + MFCC delta (13) + prosody (3) = 96 维
            # 如果需要 80 维,只使用 Mel 频谱
            mfcc_delta = librosa.feature.delta(mfcc)  # (n_mfcc, T_frames)

            # 确保所有特征长度一致
            min_len = min(
                mel_spec.shape[1],
                mfcc_delta.shape[1],
                pitch.shape[0],
            )

            # 截断到相同长度
            mel_spec = mel_spec[:, :min_len]
            mfcc_delta = mfcc_delta[:, :min_len]
            pitch = pitch[:min_len]
            energy = energy[:min_len]
            zcr = zcr[:min_len]

            # 组合特征
            if feature_dim <= self.n_mels:
                features = mel_spec[:feature_dim].T
            else:
                combined = np.vstack([
                    mel_spec,  # (n_mels, T)
                    mfcc_delta,  # (n_mfcc, T)
                    pitch.T,  # (1, T)
                    energy.T,  # (1, T)
                    zcr.T,  # (1, T)
                ])  # (n_mels + n_mfcc + 3, T)

                # 截断或填充到目标维度
                if combined.shape[0] > feature_dim:
                    features = combined[:feature_dim].T
                else:
                    pad_size = feature_dim - combined.shape[0]
                    padded = np.pad(
                        combined,
                        ((0, pad_size), (0, 0)),
                        mode='constant'
                    )
                    features = padded.T

        return features  # (T_frames, feature_dim)

    def process_video_audio(
        self,
        video_path: str,
        feature_dim: int = 80,
        cleanup: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        从视频中提取音频特征

        Args:
            video_path: 视频文件路径
            feature_dim: 特征维度
            cleanup: 是否清理临时文件

        Returns:
            features: (T_frames, feature_dim) 音频特征
            fps: 特征帧率
        """
        # 提取音频
        audio_path = self.extract_audio_from_video(video_path)

        try:
            # 加载音频
            audio, sr = self.load_audio(audio_path)

            # 提取特征
            features = self.extract_all_features(audio, feature_dim)

            # 计算特征帧率
            fps = sr / self.hop_length

            return features, fps

        finally:
            # 清理临时文件
            if cleanup and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    def align_features_to_video(
        self,
        audio_features: np.ndarray,
        video_length: int,
    ) -> np.ndarray:
        """
        将音频特征对齐到视频帧数

        Args:
            audio_features: (T_audio, D) 音频特征
            video_length: 视频帧数

        Returns:
            aligned_features: (T_video, D) 对齐后的特征
        """
        from scipy.interpolate import interp1d

        T_audio, D = audio_features.shape

        if T_audio == video_length:
            return audio_features

        # 使用线性插值对齐
        x_audio = np.linspace(0, 1, T_audio)
        x_video = np.linspace(0, 1, video_length)

        aligned_features = np.zeros((video_length, D), dtype=np.float32)

        for d in range(D):
            f = interp1d(x_audio, audio_features[:, d], kind='linear')
            aligned_features[:, d] = f(x_video)

        return aligned_features
