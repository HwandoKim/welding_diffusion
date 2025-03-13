import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import torch
import os
import pandas as pd
import re
from tqdm import tqdm

def load_csv_point_cloud(file_path):
    """CSV 파일에서 포인트 클라우드 데이터 로드 (x,y,z,f,scl1 포맷)"""
    try:
        df = pd.read_csv(file_path)
        # 필요한 열이 있는지 확인
        required_columns = ['x', 'y', 'z', 'f', 'scl1']
        for col in required_columns:
            if col not in df.columns:
                print(f"경고: {file_path}에 '{col}' 열이 없습니다.")
                
        # x, y, z 좌표만 추출하여 numpy 배열로 변환
        points_xyz = df[['x', 'y', 'z']].values
        
        # 추가 특성도 함께 저장
        features = df[['f', 'scl1']].values if 'f' in df.columns and 'scl1' in df.columns else None
        
        return points_xyz, features
    except Exception as e:
        print(f"파일 {file_path} 로딩 중 오류 발생: {e}")
        return None, None

def visualize_point_cloud(points, features=None, title="Point Cloud"):
    """Open3D를 사용하여 3D 포인트 클라우드 시각화
    features가 제공되면 첫 번째 특성(f)을 색상으로 표현"""
    if points is None or len(points) == 0:
        print("시각화할 포인트가 없습니다.")
        return
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 특성 값이 제공된 경우 색상 매핑
    if features is not None and features.shape[1] >= 1:
        # f 값을 기준으로 색상 매핑 (최소값: 파란색, 최대값: 빨간색)
        f_values = features[:, 0]
        if np.max(f_values) > np.min(f_values):  # 0으로 나누기 방지
            normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
        else:
            normalized_f = np.zeros_like(f_values)
        
        # 색상 생성: 파란색(0,0,1) -> 빨간색(1,0,0)
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = normalized_f  # R 채널
        colors[:, 2] = 1 - normalized_f  # B 채널
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 특성이 없는 경우 랜덤 컬러 설정
        colors = np.random.uniform(0, 1, size=(points.shape[0], 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 시각화
    o3d.visualization.draw_geometries([pcd], window_name=title, width=800, height=600)

def save_point_cloud_as_ply(points, features=None, filename="point_cloud.ply"):
    """포인트 클라우드를 PLY 파일로 저장 (특성 포함)"""
    if points is None or len(points) == 0:
        print(f"저장할 포인트가 없습니다: {filename}")
        return
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 특성 값이 제공된 경우 색상 매핑
    if features is not None and features.shape[1] >= 1:
        f_values = features[:, 0]
        if np.max(f_values) > np.min(f_values):  # 0으로 나누기 방지
            normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
        else:
            normalized_f = np.zeros_like(f_values)
        
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = normalized_f  # R 채널
        colors[:, 2] = 1 - normalized_f  # B 채널
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")

def extract_frame_number(filename):
    """파일 이름에서 프레임 번호 추출 (filtered_frame_101.csv -> 101)"""
    match = re.search(r'filtered_frame_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    return None

def process_sequential_frames(data_dir, output_dir=None, start_frame=101, end_frame=200):
    """연속된 프레임 CSV 파일 처리 (filtered_frame_101.csv부터 filtered_frame_200.csv)"""
    if output_dir is None:
        output_dir = os.path.join(data_dir, "visualized")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ply"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "png"), exist_ok=True)
    
    # 모든 프레임 처리
    for frame_num in tqdm(range(start_frame, end_frame + 1), desc="프레임 처리 중"):
        csv_filename = f"filtered_frame_{frame_num}.csv"
        csv_path = os.path.join(data_dir, csv_filename)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(csv_path):
            print(f"경고: {csv_path} 파일이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        # CSV 파일 로드
        points, features = load_csv_point_cloud(csv_path)
        if points is None:
            continue
            
        # PLY로 저장
        ply_filename = os.path.join(output_dir, "ply", f"frame_{frame_num}.ply")
        save_point_cloud_as_ply(points, features, ply_filename)
        
        # 미니어처 시각화 생성
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        if features is not None:
            f_values = features[:, 0]
            if np.max(f_values) > np.min(f_values):
                normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
            else:
                normalized_f = np.zeros_like(f_values)
            colors = plt.cm.coolwarm(normalized_f)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
            
        ax.set_title(f"Frame {frame_num}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "png", f"frame_{frame_num}.png"), dpi=150)
        plt.close()

def create_dataset_from_csv_frames(data_dir, output_npz, start_frame=101, end_frame=200):
    """연속된 프레임 CSV 파일을 처리하여 학습용 NPZ 데이터셋 생성"""
    all_points = []
    all_features = []
    
    for frame_num in tqdm(range(start_frame, end_frame + 1), desc="데이터셋 생성 중"):
        csv_filename = f"filtered_frame_{frame_num}.csv"
        csv_path = os.path.join(data_dir, csv_filename)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(csv_path):
            print(f"경고: {csv_path} 파일이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        # CSV 파일 로드
        points, features = load_csv_point_cloud(csv_path)
        if points is None:
            continue
            
        all_points.append(points)
        if features is not None:
            all_features.append(features)
    
    # 데이터셋 저장
    if all_points:
        np.savez(output_npz, 
                 points=np.array(all_points),
                 features=np.array(all_features) if all_features else None)
        print(f"데이터셋이 {output_npz}에 저장되었습니다.")
    else:
        print("처리할 데이터가 없습니다.")

def analyze_frame_statistics(data_dir, start_frame=101, end_frame=200):
    """모든 프레임의 통계 정보 분석"""
    point_counts = []
    feature_stats = {'f': [], 'scl1': []}
    
    for frame_num in tqdm(range(start_frame, end_frame + 1), desc="프레임 통계 분석 중"):
        csv_filename = f"filtered_frame_{frame_num}.csv"
        csv_path = os.path.join(data_dir, csv_filename)
        
        # 파일이 존재하는지 확인
        if not os.path.exists(csv_path):
            continue
        
        # CSV 파일 로드 (직접 DataFrame으로 분석)
        try:
            df = pd.read_csv(csv_path)
            
            # 포인트 수
            point_counts.append(len(df))
            
            # 특성 통계
            if 'f' in df.columns:
                feature_stats['f'].append({
                    'min': df['f'].min(),
                    'max': df['f'].max(),
                    'mean': df['f'].mean(),
                    'std': df['f'].std()
                })
            
            if 'scl1' in df.columns:
                feature_stats['scl1'].append({
                    'min': df['scl1'].min(),
                    'max': df['scl1'].max(),
                    'mean': df['scl1'].mean(),
                    'std': df['scl1'].std()
                })
        except Exception as e:
            print(f"파일 {csv_path} 분석 중 오류 발생: {e}")
    
    # 통계 요약
    if point_counts:
        print("\n===== 프레임 통계 요약 =====")
        print(f"총 처리된 프레임: {len(point_counts)}")
        print(f"포인트 수 - 평균: {np.mean(point_counts):.2f}, 최소: {np.min(point_counts)}, 최대: {np.max(point_counts)}")
        
        if feature_stats['f']:
            f_min = np.min([stat['min'] for stat in feature_stats['f']])
            f_max = np.max([stat['max'] for stat in feature_stats['f']])
            f_mean = np.mean([stat['mean'] for stat in feature_stats['f']])
            print(f"특성 'f' - 범위: [{f_min:.4f}, {f_max:.4f}], 평균: {f_mean:.4f}")
        
        if feature_stats['scl1']:
            scl1_min = np.min([stat['min'] for stat in feature_stats['scl1']])
            scl1_max = np.max([stat['max'] for stat in feature_stats['scl1']])
            scl1_mean = np.mean([stat['mean'] for stat in feature_stats['scl1']])
            print(f"특성 'scl1' - 범위: [{scl1_min:.4f}, {scl1_max:.4f}], 평균: {scl1_mean:.4f}")
    else:
        print("처리된 프레임이 없습니다.")

def create_animation_from_frames(png_dir, output_gif, fps=10):
    """PNG 이미지 시퀀스로부터 애니메이션 GIF 생성"""
    try:
        import imageio
        
        # 프레임 번호순으로 PNG 파일 정렬
        png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')], 
                          key=lambda x: int(re.search(r'frame_(\d+)\.png', x).group(1)))
        
        # 모든 이미지 로드
        images = []
        for png_file in png_files:
            images.append(imageio.imread(os.path.join(png_dir, png_file)))
        
        # GIF 생성
        imageio.mimsave(output_gif, images, fps=fps)
        print(f"애니메이션이 {output_gif}에 저장되었습니다.")
    except ImportError:
        print("imageio 라이브러리가 필요합니다. pip install imageio로 설치하세요.")
    except Exception as e:
        print(f"애니메이션 생성 중 오류 발생: {e}")

def chamfer_distance(x, y):
    """두 포인트 클라우드 간의 Chamfer Distance 계산 (xyz 좌표만 사용)"""
    # x: [batch_size, num_points, 3] or [num_points, 3]
    # y: [batch_size, num_points, 3] or [num_points, 3]
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # 포인트 클라우드 형태 통일
    if x.ndim == 2:  # 단일 포인트 클라우드인 경우 [num_points, 3]
        x = x.reshape(1, -1, 3)
    if y.ndim == 2:
        y = y.reshape(1, -1, 3)
    
    batch_size = x.shape[0]
    total_cd = 0.0
    
    for i in range(batch_size):
        x_points = x[i]
        y_points = y[i]
        
        # x -> y 방향 거리
        x_to_y = pairwise_distances(x_points, y_points, metric='euclidean')
        x_to_y_min = np.min(x_to_y, axis=1).mean()
        
        # y -> x 방향 거리
        y_to_x = pairwise_distances(y_points, x_points, metric='euclidean')
        y_to_x_min = np.min(y_to_x, axis=1).mean()
        
        cd = (x_to_y_min + y_to_x_min) / 2.0
        total_cd += cd
    
    return total_cd / batch_size

def feature_mse(x_features, y_features):
    """두 포인트 클라우드의 특성(f, scl1) 간의 MSE 계산"""
    if isinstance(x_features, torch.Tensor):
        x_features = x_features.detach().cpu().numpy()
    if isinstance(y_features, torch.Tensor):
        y_features = y_features.detach().cpu().numpy()
    
    # 포인트 클라우드 형태 통일
    if x_features.ndim == 2:  # [num_points, 2]
        x_features = x_features.reshape(1, -1, 2)
    if y_features.ndim == 2:
        y_features = y_features.reshape(1, -1, 2)
    
    batch_size = x_features.shape[0]
    total_mse = 0.0
    
    for i in range(batch_size):
        x_feat = x_features[i]
        y_feat = y_features[i]
        
        # 특성별 MSE 계산
        mse = np.mean((x_feat - y_feat) ** 2)
        total_mse += mse
    
    return total_mse / batch_size

def compare_frames(data_dir, frame1, frame2, visualize=True):
    """두 프레임 간의 비교 분석"""
    # 두 프레임 로드
    frame1_path = os.path.join(data_dir, f"filtered_frame_{frame1}.csv")
    frame2_path = os.path.join(data_dir, f"filtered_frame_{frame2}.csv")
    
    points1, features1 = load_csv_point_cloud(frame1_path)
    points2, features2 = load_csv_point_cloud(frame2_path)
    
    if points1 is None or points2 is None:
        print("하나 이상의 프레임을 로드할 수 없습니다.")
        return
    
    # 두 프레임 간의 Chamfer Distance 계산
    cd = chamfer_distance(points1, points2)
    print(f"프레임 {frame1}와 {frame2} 간의 Chamfer Distance: {cd:.6f}")
    
    # 두 프레임의 특성 비교
    if features1 is not None and features2 is not None:
        feat_mse = feature_mse(features1, features2)
        print(f"프레임 {frame1}와 {frame2} 간의 특성 MSE: {feat_mse:.6f}")
    
    # 시각화 (선택 사항)
    if visualize:
        fig = plt.figure(figsize=(12, 6))
        
        # 프레임 1
        ax1 = fig.add_subplot(121, projection='3d')
        if features1 is not None:
            f_values = features1[:, 0]
            if np.max(f_values) > np.min(f_values):
                normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
            else:
                normalized_f = np.zeros_like(f_values)
            colors = plt.cm.coolwarm(normalized_f)
            ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=1, c=colors)
        else:
            ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=1)
        ax1.set_title(f"Frame {frame1}")
        ax1.set_axis_off()
        
        # 프레임 2
        ax2 = fig.add_subplot(122, projection='3d')
        if features2 is not None:
            f_values = features2[:, 0]
            if np.max(f_values) > np.min(f_values):
                normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
            else:
                normalized_f = np.zeros_like(f_values)
            colors = plt.cm.coolwarm(normalized_f)
            ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=1, c=colors)
        else:
            ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=1)
        ax2.set_title(f"Frame {frame2}")
        ax2.set_axis_off()
        
        plt.suptitle(f"프레임 비교: CD={cd:.6f}")
        plt.tight_layout()
        plt.show()

def main():
    """샘플 사용법 예시"""
    # 예시 경로
    data_dir = "/workspace/welding_paper_project/data/bead_all_data"  # CSV 파일이 있는 디렉토리
    output_dir = "/workspace/welding_paper_project/data/result"  # 결과물 저장 디렉토리
    
    # 프레임 번호 범위 설정
    start_frame = 101
    end_frame = 200
    
    # 1. 모든 프레임 처리 및 시각화
    process_sequential_frames(data_dir, output_dir, start_frame, end_frame)
    
    # 2. 데이터셋 생성
    create_dataset_from_csv_frames(data_dir, os.path.join(output_dir, "point_cloud_dataset.npz"), start_frame, end_frame)
    
    # 3. 프레임 통계 분석
    analyze_frame_statistics(data_dir, start_frame, end_frame)
    
    # 4. 애니메이션 생성 (선택 사항)
    create_animation_from_frames(os.path.join(output_dir, "png"), os.path.join(output_dir, "animation.gif"), fps=5)
    
    # 5. 특정 프레임 비교
    compare_frames(data_dir, 101, 150)
    
    # 6. 단일 프레임 시각화 예시
    frame_path = os.path.join(data_dir, "filtered_frame_101.csv")
    if os.path.exists(frame_path):
        points, features = load_csv_point_cloud(frame_path)
        visualize_point_cloud(points, features, "Frame 101")

if __name__ == "__main__":
    main()