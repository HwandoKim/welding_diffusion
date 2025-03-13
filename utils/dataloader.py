# util/dataloader.py 또는 dataset/dataloader.py에 추가

import pandas as pd
import numpy as np
import torch

def load_csv_point_cloud(csv_path, normalize=True):
    """
    단일 CSV 파일에서 포인트 클라우드 데이터 로드
    
    Args:
        csv_path: CSV 파일 경로
        normalize: 좌표를 정규화할지 여부 (기본값: True)
        
    Returns:
        points: 포인트 좌표 [N, 3] (NumPy 배열)
        features: 포인트 특성 [N, 2] (NumPy 배열)
    """
    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_path)
        
        # 필수 칼럼 확인
        required_columns = ['x', 'y', 'z', 'f', 'scl1']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV 파일에 필요한 칼럼이 없습니다: {col}")
        
        # 좌표와 특성 추출
        points = df[['x', 'y', 'z']].values.astype(np.float32)
        features = df[['f', 'scl1']].values.astype(np.float32)
        
        # 정규화 (단위 구체로)
        if normalize:
            centroid = np.mean(points, axis=0)
            points = points - centroid
            furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
            if furthest_distance > 0:  # 0으로 나누기 방지
                points = points / furthest_distance
    
    except Exception as e:
        print(f"CSV 파일 {csv_path} 로드 중 오류 발생: {e}")
        # 오류 시 빈 데이터 반환
        points = np.zeros((1, 3), dtype=np.float32)
        features = np.zeros((1, 2), dtype=np.float32)
    
    return points, features

def save_point_cloud_to_csv(points, features, csv_path):
    """
    포인트 클라우드 데이터를 CSV 파일로 저장
    
    Args:
        points: 포인트 좌표 [N, 3] (NumPy 배열 또는 텐서)
        features: 포인트 특성 [N, 2] (NumPy 배열 또는 텐서)
        csv_path: 저장할 CSV 파일 경로
    """
    # 텐서를 NumPy 배열로 변환
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    # DataFrame 생성
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'f': features[:, 0],
        'scl1': features[:, 1]
    })
    
    # CSV로 저장
    df.to_csv(csv_path, index=False)
    print(f"포인트 클라우드가 {csv_path}에 저장되었습니다")

# 테스트 코드
if __name__ == "__main__":
    import os
    
    # 테스트 CSV 파일 경로
    csv_path = "../data/sample.csv"  # 실제 경로로 수정
    
    if os.path.exists(csv_path):
        points, features = load_csv_point_cloud(csv_path)
        print(f"로드된 포인트 수: {len(points)}")
        print(f"포인트 샘플: {points[:5]}")
        print(f"특성 샘플: {features[:5]}")
        
        # 로드한 데이터 다시 저장 테스트
        save_path = "../data/test_save.csv"
        save_point_cloud_to_csv(points, features, save_path)