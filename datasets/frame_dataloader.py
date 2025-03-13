import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class FramePointCloudDataset(Dataset):
    """
    프레임별 포인트 클라우드 데이터셋 클래스
    항상 연속된 프레임으로 작동하는 데이터셋
    """
    def __init__(self, data_dir, min_frame=100, max_frame=200, transform=None, normalize=True):
        self.data_dir = data_dir
        self.min_frame = min_frame
        self.max_frame = max_frame
        self.transform = transform
        self.normalize = normalize
        
        # 디버깅을 위한 로그 추가
        print(f"데이터 디렉토리: {os.path.abspath(data_dir)}")
        print(f"디렉토리 존재 여부: {os.path.exists(data_dir)}")
        
        # 프레임 번호에 맞는 파일 찾기
        self.frame_files = {}
        found_frames = []
        
        # 데이터 디렉토리의 모든 하위 디렉토리 검색
        for root, dirs, files in os.walk(data_dir):
            for file_name in files:
                if file_name.endswith('.csv'):
                    full_path = os.path.join(root, file_name)
                    print(f"CSV 파일 발견: {full_path}")
                    
                    try:
                        # 파일 이름에서 'filtered_frame_XXX.csv' 패턴 찾기
                        if 'filtered_frame_' in file_name:
                            frame_num = int(file_name.split('filtered_frame_')[1].split('.')[0])
                            found_frames.append(frame_num)
                            
                            if min_frame <= frame_num <= max_frame:
                                self.frame_files[frame_num] = full_path
                                print(f"프레임 {frame_num} 추가됨: {full_path}")
                    except (ValueError, IndexError) as e:
                        print(f"파일 파싱 오류: {file_name} - {str(e)}")
                        continue
        
        # 디버깅 정보 출력
        print(f"발견한 프레임 번호: {sorted(found_frames)}")
        print(f"필터링된 프레임 번호 ({min_frame}~{max_frame}): {sorted(self.frame_files.keys())}")
        
        # 사용 가능한 프레임 번호 목록
        self.frame_numbers = sorted(self.frame_files.keys())
        
        print(f"프레임 {min_frame}부터 {max_frame}까지 총 {len(self.frame_numbers)}개의 프레임 로드됨")
    
    def __len__(self):
        """데이터셋의 샘플 수 반환 (가이던스 사용 시 마지막 프레임은 제외)"""
        length = len(self.frame_numbers) - 1 if len(self.frame_numbers) > 0 else 0
        return max(0, length)  # 항상 0 이상의 값 반환
    
    def normalize_point_cloud(self, points):
        # 중심점 계산
        centroid = np.mean(points, axis=0)
        # 중심점을 원점으로 이동
        centered_points = points - centroid
        
        # 최대 거리 계산
        max_dist = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
        # 0으로 나누기 방지
        if max_dist < 1e-10:
            print("경고: 최대 거리가 너무 작음, 정규화를 건너뜁니다.")
            return points
            
        # [-1, 1] 범위로 스케일링
        normalized_points = centered_points / max_dist
        
        return normalized_points
    
    def __getitem__(self, idx):
        if len(self.frame_numbers) <= idx + 1:
            raise IndexError(f"인덱스 {idx}는 유효하지 않습니다. 프레임 개수: {len(self.frame_numbers)}")
            
        current_frame = self.frame_numbers[idx]
        next_frame = self.frame_numbers[idx + 1]
        
        print(f"데이터 로드: 현재 프레임 {current_frame}, 다음 프레임 {next_frame}")
        
        # 현재 프레임 데이터 로드
        current_file = self.frame_files[current_frame]
        current_data = self._load_point_cloud(current_file, current_frame)
        
        # 다음 프레임 데이터 로드
        next_file = self.frame_files[next_frame]
        next_data = self._load_point_cloud(next_file, next_frame)
        
        # 현재 및 다음 프레임 데이터 결합
        data = {
            'current_points': current_data['points'],          # 현재 프레임 포인트
            'current_features': current_data['features'],      # 현재 프레임 특성
            'current_frame': current_data['frame'],            # 현재 프레임 번호
            'next_points': next_data['points'],                # 다음 프레임 포인트
            'next_features': next_data['features'],            # 다음 프레임 특성
            'next_frame': next_data['frame'],                  # 다음 프레임 번호
            'num_points': current_data['num_points']           # 포인트 수
        }
        
        return data
    
    def _load_point_cloud(self, file_path, frame_num):
        print(f"파일 로드 시도: {file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
            
        try:
            # 포인트 클라우드 데이터 읽기
            point_cloud_df = pd.read_csv(file_path)
            
            # 컬럼 확인
            print(f"파일 {os.path.basename(file_path)}의 컬럼: {point_cloud_df.columns.tolist()}")
            
            # 필요한 컬럼 존재 확인
            required_cols = ['x', 'y', 'z']
            for col in required_cols:
                if col not in point_cloud_df.columns:
                    raise ValueError(f"필수 컬럼 '{col}'이 파일에 없습니다.")
            
            # 특성 컬럼 확인 및 처리
            features_cols = []
            if 'scl1' in point_cloud_df.columns:
                features_cols.append('scl1')
            if 'f' in point_cloud_df.columns:
                features_cols.append('f')
                
            # 필요한 컬럼 추출
            points = point_cloud_df[['x', 'y', 'z']].values.astype(np.float32)
            
            # 특성 컬럼이 있으면 사용, 없으면 0으로 채움
            if features_cols:
                features = point_cloud_df[features_cols].values.astype(np.float32)
            else:
                # 특성이 없으면 0으로 채운 배열 생성
                features = np.zeros((len(points), 1), dtype=np.float32)
                print(f"경고: 특성 컬럼이 없어 0으로 채웁니다.")
            
            print(f"로드된 포인트 수: {len(points)}, 특성 크기: {features.shape}")
            
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
                'points': points_tensor,          # [N, 3] 텐서, N은 포인트 수
                'features': features_tensor,      # [N, F] 텐서, F는 특성 수
                'frame': torch.tensor(frame_num, dtype=torch.long),  # 프레임 번호
                'num_points': len(points)
            }
            
            print(f"데이터 로드 성공: 프레임 {frame_num}, 포인트 수 {len(points)}")
            return data
            
        except Exception as e:
            print(f"파일 {file_path} 로드 중 오류 발생: {str(e)}")
            raise


def get_dataloader(config):
    """
    데이터로더 생성 함수
    """
    # 데이터 디렉토리 확인
    data_dir = config.get('data_dir', 'data')
    print(f"데이터 디렉토리 설정: {data_dir}")
    print(f"디렉토리 절대 경로: {os.path.abspath(data_dir)}")
    
    # 설정 로그
    print("데이터로더 설정:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    try:
        # 데이터셋 생성
        dataset = FramePointCloudDataset(
            data_dir=data_dir,
            min_frame=config.get('min_frame', 100),
            max_frame=config.get('max_frame', 200),
            transform=None,
            normalize=config.get('normalize', True)
        )
        
        # 데이터셋의 평균 포인트 수 계산
        if len(dataset) > 0:
            total_points = 0
            num_samples = min(10, len(dataset))
            for i in range(num_samples):
                print(f"샘플 {i} 로드 중...")
                data = dataset[i]
                total_points += data['num_points']
            avg_points = int(total_points / num_samples)
            print(f"평균 포인트 수: {avg_points}")
        else:
            avg_points = 0
            print("데이터셋이 비어 있습니다.")
        
        # 데이터로더 생성 (배치 사이즈 1로 고정)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # 배치 사이즈를 1로 고정
            shuffle=config.get('shuffle', False),  # 프레임 순서 유지를 위해 셔플 비활성화
            num_workers=config.get('num_workers', 4),
            pin_memory=config.get('pin_memory', True),
            drop_last=False
        )
        
        print(f"프레임 기반 데이터로더 생성 완료. 총 {len(dataset)}개의 프레임 쌍.")
        
        return dataloader, avg_points
    
    except Exception as e:
        print(f"데이터로더 생성 중 오류 발생: {str(e)}")
        # 스택 트레이스 출력
        import traceback
        traceback.print_exc()
        
        # 빈 데이터로더 반환
        print("빈 데이터셋으로 대체합니다.")
        empty_dataset = FramePointCloudDataset(
            data_dir=data_dir,
            min_frame=999999,  # 존재하지 않는 프레임 번호로 설정
            max_frame=999999
        )
        empty_loader = DataLoader(empty_dataset, batch_size=1)
        return empty_loader, 0


# 메인 실행 코드
if __name__ == "__main__":
    print("데이터로더 테스트 스크립트 실행")
    
    # 설정 생성
    class Config(dict):
        def __init__(self, *args, **kwargs):
            super(Config, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    # 실제 파일이 있는 경로로 조정
    config = Config({
        'data_dir': '/workspace/welding_paper_project/data',  # 실제 데이터 경로
        'min_frame': 101,      # 실제 시작 프레임
        'max_frame': 200,      # 실제 끝 프레임
        'num_workers': 0,      # 디버깅을 위해 워커 수 0으로 설정
        'pin_memory': False,
        'shuffle': False,
        'normalize': True
    })
    
    print("\n" + "="*50)
    print("현재 작업 디렉토리:", os.getcwd())
    print("="*50 + "\n")
    
    try:
        # 데이터로더 생성
        print("\n데이터로더 생성 시도...")
        dataloader, avg_points = get_dataloader(config)
        print(f"데이터로더 생성 완료: {len(dataloader)}개 배치, 평균 {avg_points}개 포인트")
        
        # 첫 번째 배치 확인
        if len(dataloader) > 0:
            print("\n첫 번째 배치 테스트:")
            batch = next(iter(dataloader))
            
            print("배치 내용:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: 형태 {v.shape}, 타입 {v.dtype}")
                else:
                    print(f"  {k}: {v}")
                    
            # 텐서 통계 출력
            if 'current_points' in batch:
                points = batch['current_points'][0]  # 첫 번째 배치 아이템
                print(f"\n포인트 통계:")
                print(f"  최소값: {points.min(dim=0)[0]}")
                print(f"  최대값: {points.max(dim=0)[0]}")
                print(f"  평균: {points.mean(dim=0)}")
                print(f"  표준편차: {points.std(dim=0)}")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()