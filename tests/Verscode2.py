import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm

class PointCloudCSVDataset(Dataset):
    """연속된 프레임 CSV 파일을 처리하는 데이터셋 클래스"""
    
    def __init__(self, data_dir, start_frame=101, end_frame=200, num_points=2048, transform=None):
        """
        Args:
            data_dir (str): CSV 파일이 있는 디렉토리 경로
            start_frame (int): 시작 프레임 번호
            end_frame (int): 종료 프레임 번호
            num_points (int): 사용할 포인트 수 (샘플링 또는 패딩)
            transform (callable, optional): 샘플에 적용할 변환
        """
        self.data_dir = data_dir
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.num_points = num_points
        self.transform = transform
        
        # 유효한 프레임 목록 생성
        self.valid_frames = []
        for frame_num in range(start_frame, end_frame + 1):
            csv_path = os.path.join(data_dir, f"filtered_frame_{frame_num}.csv")
            if os.path.exists(csv_path):
                self.valid_frames.append(frame_num)
        
        print(f"총 {len(self.valid_frames)}개의 유효한 프레임을 찾았습니다.")
        
        # 통계 정보 계산
        self._compute_statistics()
    
    def _compute_statistics(self):
        """데이터셋 통계 계산 (정규화에 사용)"""
        print("데이터셋 통계 계산 중...")
        
        # 샘플 파일 검사 (처음 5개 프레임)
        sample_frames = self.valid_frames[:min(5, len(self.valid_frames))]
        all_features = []
        
        for frame_num in sample_frames:
            csv_path = os.path.join(self.data_dir, f"filtered_frame_{frame_num}.csv")
            try:
                df = pd.read_csv(csv_path)
                if 'f' in df.columns and 'scl1' in df.columns:
                    features = df[['f', 'scl1']].values
                    all_features.append(features)
            except Exception as e:
                print(f"경고: 파일 {csv_path} 통계 계산 중 오류 발생: {e}")
        
        # 특성 통계 계산
        if all_features:
            all_features = np.vstack(all_features)
            self.feature_min = np.min(all_features, axis=0)
            self.feature_max = np.max(all_features, axis=0)
            self.feature_mean = np.mean(all_features, axis=0)
            self.feature_std = np.std(all_features, axis=0)
            
            print(f"특성 범위 - f: [{self.feature_min[0]:.4f}, {self.feature_max[0]:.4f}], " +
                  f"scl1: [{self.feature_min[1]:.4f}, {self.feature_max[1]:.4f}]")
        else:
            self.feature_min = np.array([0, 0])
            self.feature_max = np.array([1, 1])
            self.feature_mean = np.array([0.5, 0.5])
            self.feature_std = np.array([0.5, 0.5])
            
            print("경고: 특성 통계를 계산할 수 없습니다. 기본값을 사용합니다.")
    
    def __len__(self):
        return len(self.valid_frames)
    
    def __getitem__(self, idx):
        frame_num = self.valid_frames[idx]
        csv_path = os.path.join(self.data_dir, f"filtered_frame_{frame_num}.csv")
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_path)
            
            # 좌표와 특성 추출
            points = df[['x', 'y', 'z']].values
            features = df[['f', 'scl1']].values if 'f' in df.columns and 'scl1' in df.columns else None
            
            # 포인트 수 조정
            if len(points) > self.num_points:
                # 랜덤 샘플링
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
                if features is not None:
                    features = features[indices]
            elif len(points) < self.num_points:
                # 부족한 포인트 복제
                indices = np.random.choice(len(points), self.num_points - len(points), replace=True)
                extra_points = points[indices]
                points = np.vstack([points, extra_points])
                
                if features is not None:
                    extra_features = features[indices]
                    features = np.vstack([features, extra_features])
            
            # 특성 정규화 (0-1 범위로 스케일링)
            if features is not None:
                # 최대-최소 정규화
                features = (features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
                # 이상치 클리핑
                features = np.clip(features, 0, 1)
            else:
                # 특성 데이터가 없는 경우 0으로 채움
                features = np.zeros((self.num_points, 2))
            
            # 변환 적용 (필요시)
            if self.transform:
                points, features = self.transform(points, features)
            
            # torch 텐서로 변환
            points_tensor = torch.FloatTensor(points)
            features_tensor = torch.FloatTensor(features)
            
            return points_tensor, features_tensor
        
        except Exception as e:
            print(f"오류: 파일 {csv_path} 처리 중 {e}")
            # 오류 발생 시 빈 데이터 반환
            points = np.zeros((self.num_points, 3))
            features = np.zeros((self.num_points, 2))
            return torch.FloatTensor(points), torch.FloatTensor(features)

class PointCloudTransforms:
    """포인트 클라우드 데이터에 적용할 수 있는 변환 모음"""
    
    @staticmethod
    def random_rotate(points, features, axis=None):
        """
        포인트 클라우드를 랜덤하게 회전 (특성은 변경 없음)
        
        Args:
            points (np.ndarray): 포인트 좌표 [N, 3]
            features (np.ndarray): 포인트 특성 [N, 2]
            axis (int, optional): 회전축 (0=x, 1=y, 2=z, None=모든 축)
            
        Returns:
            tuple: (회전된 포인트, 원본 특성)
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        
        if axis is None:
            # 모든 축에 대한 랜덤 회전
            axis = np.random.randint(3)
        
        if axis == 0:  # X축 회전
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cosval, -sinval],
                [0, sinval, cosval]
            ])
        elif axis == 1:  # Y축 회전
            rotation_matrix = np.array([
                [cosval, 0, sinval],
                [0, 1, 0],
                [-sinval, 0, cosval]
            ])
        else:  # Z축 회전
            rotation_matrix = np.array([
                [cosval, -sinval, 0],
                [sinval, cosval, 0],
                [0, 0, 1]
            ])
        
        # 포인트 클라우드 회전
        rotated_points = np.dot(points, rotation_matrix)
        
        return rotated_points, features
    
    @staticmethod
    def jitter_points(points, features, sigma=0.01, clip=0.05):
        """
        포인트 클라우드에 가우시안 노이즈 추가 (특성은 변경 없음)
        
        Args:
            points (np.ndarray): 포인트 좌표 [N, 3]
            features (np.ndarray): 포인트 특성 [N, 2]
            sigma (float): 노이즈 표준편차
            clip (float): 노이즈 클리핑 범위
            
        Returns:
            tuple: (노이즈가 추가된 포인트, 원본 특성)
        """
        assert(clip > 0)
        jittered_points = np.clip(sigma * np.random.randn(*points.shape), -1*clip, clip)
        jittered_points += points
        
        return jittered_points, features
    
    @staticmethod
    def random_scale(points, features, scale_low=0.8, scale_high=1.2):
        """
        포인트 클라우드를 랜덤하게 스케일링 (특성은 변경 없음)
        
        Args:
            points (np.ndarray): 포인트 좌표 [N, 3]
            features (np.ndarray): 포인트 특성 [N, 2]
            scale_low (float): 최소 스케일 계수
            scale_high (float): 최대 스케일 계수
            
        Returns:
            tuple: (스케일링된 포인트, 원본 특성)
        """
        scale = np.random.uniform(scale_low, scale_high)
        scaled_points = points * scale
        
        return scaled_points, features
    
    @staticmethod
    def random_shift(points, features, shift_range=0.1):
        """
        포인트 클라우드를 랜덤하게 이동 (특성은 변경 없음)
        
        Args:
            points (np.ndarray): 포인트 좌표 [N, 3]
            features (np.ndarray): 포인트 특성 [N, 2]
            shift_range (float): 이동 범위
            
        Returns:
            tuple: (이동된 포인트, 원본 특성)
        """
        shifts = np.random.uniform(-shift_range, shift_range, size=(1, 3))
        shifted_points = points + shifts
        
        return shifted_points, features
    
    @staticmethod
    def normalize_points(points, features):
        """
        포인트 클라우드를 정규화 (중심이 원점, 최대 거리 1)
        
        Args:
            points (np.ndarray): 포인트 좌표 [N, 3]
            features (np.ndarray): 포인트 특성 [N, 2]
            
        Returns:
            tuple: (정규화된 포인트, 원본 특성)
        """
        # 중심점 계산
        centroid = np.mean(points, axis=0)
        
        # 원점으로 이동
        points = points - centroid
        
        # 최대 거리 계산
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        
        # 스케일링
        if max_dist > 0:
            points = points / max_dist
        
        return points, features
    
    @staticmethod
    def random_dropout(points, features, max_dropout_ratio=0.2):
        """
        포인트 클라우드에서 랜덤하게 일부 포인트 드롭아웃 후 복제로 채움
        
        Args:
            points (np.ndarray): 포인트 좌표 [N, 3]
            features (np.ndarray): 포인트 특성 [N, 2]
            max_dropout_ratio (float): 최대 드롭아웃 비율
            
        Returns:
            tuple: (드롭아웃된 포인트, 드롭아웃된 특성)
        """
        dropout_ratio = np.random.random() * max_dropout_ratio
        drop_idx = np.where(np.random.random((points.shape[0])) <= dropout_ratio)[0]
        
        if len(drop_idx) > 0:
            # 드롭아웃될 포인트의 인덱스
            keep_idx = np.setdiff1d(np.arange(points.shape[0]), drop_idx)
            
            # 남은 포인트에서 랜덤 복제하여 원래 크기 유지
            replace_idx = np.random.choice(keep_idx, size=len(drop_idx), replace=True)
            
            # 드롭아웃 및 복제
            points[drop_idx] = points[replace_idx]
            features[drop_idx] = features[replace_idx]
        
        return points, features
    
    @staticmethod
    def compose(transforms):
        """
        여러 변환을 순차적으로 적용하는 함수
        
        Args:
            transforms (list): 적용할 변환 함수 리스트
            
        Returns:
            function: 합성된 변환 함수
        """
        def composed_transform(points, features):
            for transform in transforms:
                points, features = transform(points, features)
            return points, features
        
        return composed_transform

def create_dataloader(data_dir, batch_size=32, start_frame=101, end_frame=200, num_points=2048, apply_transforms=True):
    """
    포인트 클라우드 데이터로더 생성
    
    Args:
        data_dir (str): CSV 파일이 있는 디렉토리 경로
        batch_size (int): 배치 크기
        start_frame (int): 시작 프레임 번호
        end_frame (int): 종료 프레임 번호
        num_points (int): 사용할 포인트 수
        apply_transforms (bool): 데이터 증강 적용 여부
        
    Returns:
        DataLoader: 포인트 클라우드 데이터로더
    """
    
    # 데이터 증강 구성
    if apply_transforms:
        transform = PointCloudTransforms.compose([
            # 랜덤 회전
            lambda p, f: PointCloudTransforms.random_rotate(p, f),
            # 랜덤 지터링
            lambda p, f: PointCloudTransforms.jitter_points(p, f, sigma=0.01, clip=0.02),
            # 랜덤 스케일링
            lambda p, f: PointCloudTransforms.random_scale(p, f, scale_low=0.9, scale_high=1.1),
            # 정규화
            lambda p, f: PointCloudTransforms.normalize_points(p, f)
        ])
    else:
        # 정규화만 적용
        transform = PointCloudTransforms.normalize_points
    
    # 데이터셋 생성
    dataset = PointCloudCSVDataset(
        data_dir=data_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        num_points=num_points,
        transform=transform
    )
    
    # 데이터로더 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    return dataloader, dataset

def test_dataset(data_dir, start_frame=101, end_frame=105):
    """
    데이터셋 테스트 및 시각화 예시
    
    Args:
        data_dir (str): CSV 파일이 있는 디렉토리 경로
        start_frame (int): 테스트할 시작 프레임
        end_frame (int): 테스트할 종료 프레임
    """
    import matplotlib.pyplot as plt
    
    # 데이터셋 생성 (변환 적용)
    transform = PointCloudTransforms.compose([
        lambda p, f: PointCloudTransforms.random_rotate(p, f),
        lambda p, f: PointCloudTransforms.normalize_points(p, f)
    ])
    
    dataset = PointCloudCSVDataset(
        data_dir=data_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        num_points=1024,
        transform=transform
    )
    
    # 샘플 데이터 시각화
    fig = plt.figure(figsize=(15, 5))
    for i in range(min(3, len(dataset))):
        points, features = dataset[i]
        points = points.numpy()
        features = features.numpy()
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # f 값을 색상으로 표현
        f_values = features[:, 0]
        colors = plt.cm.coolwarm(f_values)
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        ax.set_title(f"Sample {i+1}")
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    # 데이터로더 테스트
    dataloader, _ = create_dataloader(
        data_dir,
        batch_size=2,
        start_frame=start_frame,
        end_frame=end_frame,
        num_points=1024
    )
    
    # 첫 번째 배치 테스트
    for batch_points, batch_features in dataloader:
        print(f"배치 shape - points: {batch_points.shape}, features: {batch_features.shape}")
        break
        
if __name__ == "__main__":
    # 테스트 실행
    data_dir = "/workspace/welding_paper_project/data/bead_all_data"  # CSV 파일이 있는 디렉토리
    test_dataset(data_dir)