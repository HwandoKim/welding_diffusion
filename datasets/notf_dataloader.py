import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PointCloudDataset(Dataset):
    """
    포인트 클라우드 데이터셋 클래스
    CSV 파일로부터 포인트 클라우드 데이터를 로드
    생성 모델 학습을 위한 데이터셋
    """
    def __init__(self, data_dir, transform=None, normalize=True):
        """
        데이터셋 초기화
        
        Args:
            data_dir (str): CSV 파일들이 있는 디렉토리 경로
            transform (callable, optional): 데이터에 적용할 변환
            normalize (bool): 포인트 클라우드 정규화 여부
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        
        # 모든 CSV 파일 경로 리스트 생성
        self.file_paths = []
        for file_name in sorted(os.listdir(data_dir)):
            if file_name.endswith('.csv'):
                self.file_paths.append(os.path.join(data_dir, file_name))
        
        print(f"총 {len(self.file_paths)}개의 포인트 클라우드 파일을 로드했습니다.")
    
    def __len__(self):
        """데이터셋의 샘플 수 반환"""
        return len(self.file_paths)
    
    def normalize_point_cloud(self, points):
        """
        포인트 클라우드 정규화
        중심을 원점으로 이동하고 [-1, 1] 범위로 스케일링
        
        Args:
            points (numpy.ndarray): 포인트 클라우드 좌표 [N, 3]
            
        Returns:
            numpy.ndarray: 정규화된 포인트 클라우드
        """
        # 중심점 계산
        centroid = np.mean(points, axis=0)
        # 중심점을 원점으로 이동
        centered_points = points - centroid
        
        # 최대 거리 계산
        max_dist = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
        # [-1, 1] 범위로 스케일링
        normalized_points = centered_points / max_dist
        
        return normalized_points
        
    def __getitem__(self, idx):
        """
        특정 인덱스의 포인트 클라우드 데이터 반환
        
        Args:
            idx (int): 데이터 인덱스
            
        Returns:
            dict: 포인트 클라우드 데이터와 메타데이터
        """
        # CSV 파일 로드
        file_path = self.file_paths[idx]
        file_name = os.path.basename(file_path)
        
        # 포인트 클라우드 데이터 읽기
        point_cloud_df = pd.read_csv(file_path)
        
        # 필요한 컬럼 추출
        points = point_cloud_df[['x', 'y', 'z']].values.astype(np.float32)
        features = point_cloud_df[[ 'scl1']].values.astype(np.float32)
        
        # 포인트 클라우드 정규화 (필요한 경우)
        if self.normalize:
            points = self.normalize_point_cloud(points)
        
        # 텐서로 변환
        points_tensor = torch.from_numpy(points)
        features_tensor = torch.from_numpy(features)
        
        # 변환 적용 (필요한 경우)
        if self.transform:
            points_tensor = self.transform(points_tensor)
        
        # 데이터 사전 생성
        data = {
            'points': points_tensor,         # [N, 3] 텐서, N은 포인트 수
            'features': features_tensor,     # [N, 2] 텐서
            'file_name': file_name,
            'num_points': len(points),
            'idx': idx                       # 데이터 인덱스
        }
        
        return data


def get_dataloader(config):
    """
    데이터로더 생성 함수 - 생성 모델용
    
    Args:
        config (dict): 데이터로더 설정이 포함된 설정 사전
        
    Returns:
        tuple: (DataLoader, 평균 포인트 수)
    """
    # 데이터셋 생성
    dataset = PointCloudDataset(
        data_dir=config.get('data_dir', 'data'),
        transform=None,  # 필요한 경우 여기에 변환 추가
        normalize=config.get('normalize', True)  # 기본적으로 정규화 활성화
    )
    
    # 데이터셋의 평균 포인트 수 계산 (샘플링에 사용)
    total_points = 0
    for i in range(min(10, len(dataset))):  # 처음 10개 파일만 확인해 평균 계산
        data = dataset[i]
        total_points += data['num_points']
    avg_points = int(total_points / min(10, len(dataset)))
    print(f"평균 포인트 수: {avg_points}")
    
    # 데이터로더 생성 (배치 사이즈 1로 고정)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 배치 사이즈를 1로 고정
        shuffle=config.get('shuffle', True),  # 생성 모델 학습을 위해 기본적으로 셔플
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=False
    )
    
    print(f"생성 모델용 데이터로더 생성 완료. 총 {len(dataset)}개의 포인트 클라우드.")
    
    return dataloader, avg_points


# 생성 모델을 위한 단일 데이터로더만 사용하므로 get_dataloaders 함수는 필요 없음


if __name__ == "__main__":
    # 데이터로더 테스트
    config = {
        'data_dir': '/workspace/welding_paper_project/data/bead_all_data',
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': True  # 생성 모델 학습을 위해 데이터 셔플
    }
    
    # 데이터로더 테스트
    dataloader = get_dataloader(config)
    print(f"데이터로더 샘플 수: {len(dataloader)}")
    
    # 첫 번째 샘플 확인
    for batch in dataloader:
        print(f"배치 크기: {batch['points'].shape}")
        print(f"포인트 수: {batch['num_points']}")
        print(f"파일 이름: {batch['file_name']}")
        break