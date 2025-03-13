import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 사용자 정의 모듈 가져오기 (위에서 정의한 코드)
# from lion_model import LION, Config  # lion-model.py 파일에서 정의한 모델
# from lion_dataset import create_dataloader, PointCloudCSVDataset  # lion-dataset.py 파일에서 정의한 데이터셋

class LIONTrainer:
    """LION (Latent Point Diffusion Model) 학습기"""
    
    def __init__(self, config, data_dir, model_dir="checkpoints"):
        """
        Args:
            config: 모델 및 학습 설정
            data_dir: CSV 파일이 있는 디렉토리 경로
            model_dir: 모델 체크포인트 저장 디렉토리
        """
        self.config = config
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # 모델 초기화
        self.model = LION(config).to(config.device)
        
        # 옵티마이저 설정
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=1e-5
        )
        
        # 학습률 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.1
        )
        
        # 체크포인트 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 로깅을 위한 통계
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
    
    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        
        for points, features in tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}"):
            self.optimizer.zero_grad()
            
            # 데이터를 장치로 이동
            points = points.to(self.config.device)
            features = features.to(self.config.device)
            
            # Encoder를 통해 latent 생성
            latent = self.model.encode(points, features)
            
            # 랜덤 timestep 선택
            batch_size = latent.shape[0]
            t = torch.randint(0, self.config.timesteps, (batch_size,), device=self.config.device).long()
            
            # Diffusion 손실 계산
            loss = self.model.p_losses(latent, t)
            
            # 역전파 및 최적화
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 에포크 평균 손실
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """검증 세트에서 모델 평가"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for points, features in tqdm(val_loader, desc="Validating"):
                # 데이터를 장치로 이동
                points = points.to(self.config.device)
                features = features.to(self.config.device)
                
                # Encoder를 통해 latent 생성
                latent = self.model.encode(points, features)
                
                # 고정된 timestep 분포로 평가
                batch_size = latent.shape[0]
                t_steps = torch.linspace(0, self.config.timesteps-1, steps=5).long()
                t_steps = t_steps[torch.randint(0, len(t_steps), (batch_size,))].to(self.config.device)
                
                # Diffusion 손실 계산
                loss = self.model.p_losses(latent, t_steps)
                
                total_loss += loss.item()
        
        # 검증 세트 평균 손실
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, start_frame=101, end_frame=200, val_ratio=0.1, batch_size=None):
        """전체 학습 과정"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # 전체 프레임 범위
        frame_range = end_frame - start_frame + 1
        
        # 학습/검증 분할
        val_frames = int(frame_range * val_ratio)
        train_end_frame = end_frame - val_frames
        val_start_frame = train_end_frame + 1
        
        print(f"학습 프레임: {start_frame}-{train_end_frame}, 검증 프레임: {val_start_frame}-{end_frame}")
        
        # 데이터로더 생성
        train_loader, _ = create_dataloader(
            self.data_dir,
            batch_size=batch_size,
            start_frame=start_frame,
            end_frame=train_end_frame,
            num_points=self.config.num_points,
            apply_transforms=True
        )
        
        val_loader, _ = create_dataloader(
            self.data_dir,
            batch_size=batch_size,
            start_frame=val_start_frame,
            end_frame=end_frame,
            num_points=self.config.num_points,
            apply_transforms=False
        )
        
        print(f"학습 배치: {len(train_loader)}, 검증 배치: {len(val_loader)}")
        
        best_val_loss = float('inf')
        
        # 학습 루프
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 한 에포크 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss = self.validate(val_loader)
            
            # 학습률 스케줄러 업데이트
            self.scheduler.step()
            
            # 현재 학습률
            lr = self.scheduler.get_last_lr()[0]
            
            # 소요 시간
            epoch_time = time.time() - start_time
            
            # 결과 출력
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {lr:.6f}, Time: {epoch_time:.2f}s")
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")
                print(f"새로운 최고 모델 저장 (Val Loss: {val_loss:.6f})")
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")
                
                # 샘플 생성 및 시각화
                self.generate_samples(4, f"samples_epoch_{epoch+1}")
    
    def save_checkpoint(self, filename):
        """모델 체크포인트 저장"""
        checkpoint_path = os.path.join(self.model_dir, filename)
        torch.save({
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
        """모델 체크포인트 로드"""
        checkpoint_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            print(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"체크포인트 로드 완료: {checkpoint_path} (에포크 {self.current_epoch})")
        return True
    
    def plot_losses(self):
        """학습 및 검증 손실 그래프 그리기"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        loss_plot_path = os.path.join(self.model_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"손실 그래프가 저장되었습니다: {loss_plot_path}")
    
    @torch.no_grad()
    def generate_samples(self, num_samples=4, output_dir=None):
        """샘플 포인트 클라우드 생성"""
        if output_dir is None:
            output_dir = os.path.join(self.model_dir, f"samples_epoch_{self.current_epoch+1}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.eval()
        
        # 샘플 생성
        points, features = self.model.generate(num_samples)
    
    # numpy 배열로 변환
    points_np = points.cpu().numpy()
    features_np = features.cpu().numpy()
    
    # 생성된 특성 역정규화 (실제 범위로 복원)
    # 참고: 실제 범위는 데이터셋 클래스에서 계산한 통계를 사용해야 합니다.
    # 여기서는 샘플로 임의의 값을 사용합니다.
    # feature_min, feature_max = dataset.feature_min, dataset.feature_max
    # features_np = features_np * (feature_max - feature_min) + feature_min
    
    # 시각화 및 저장
    fig = plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, 4)):  # 최대 4개까지만 표시
        # 포인트 클라우드 시각화
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        
        # 특성 'f'에 따라 색상 지정
        f_values = features_np[i, :, 0]
        normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values) + 1e-10)
        colors = plt.cm.coolwarm(normalized_f)
        
        ax.scatter(points_np[i, :, 0], points_np[i, :, 1], points_np[i, :, 2], s=1, c=colors)
        ax.set_title(f"Sample {i+1}")
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "samples_visualization.png"), dpi=150)
    plt.close()
    
    # 개별 파일로 저장
    for i in range(num_samples):
        # NumPy 배열로 저장
        sample_path = os.path.join(output_dir, f"sample_{i}.npz")
        np.savez(
            sample_path,
            points=points_np[i],
            features=features_np[i]
        )
        
        # CSV 형식으로도 저장
        sample_df_path = os.path.join(output_dir, f"sample_{i}.csv")
        points_sample = points_np[i]
        features_sample = features_np[i]
        
        import pandas as pd
        df = pd.DataFrame({
            'x': points_sample[:, 0],
            'y': points_sample[:, 1],
            'z': points_sample[:, 2],
            'f': features_sample[:, 0],
            'scl1': features_sample[:, 1]
        })
        
        df.to_csv(sample_df_path, index=False)
    
    # PLY 파일 저장 (Open3D 필요)
    try:
        import open3d as o3d
        
        for i in range(num_samples):
            points_sample = points_np[i]
            features_sample = features_np[i]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_sample)
            
            # 특성 'f'를 색상으로 표현
            f_values = features_sample[:, 0]
            normalized_f = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values) + 1e-10)
            
            # 파란색(0,0,1) -> 빨간색(1,0,0) 컬러맵
            colors = np.zeros((points_sample.shape[0], 3))
            colors[:, 0] = normalized_f  # R 채널
            colors[:, 2] = 1 - normalized_f  # B 채널
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            ply_path = os.path.join(output_dir, f"sample_{i}.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
    except ImportError:
        print("Open3D 라이브러리가 없어 PLY 파일은 저장하지 않습니다.")
    
    print(f"{num_samples}개의 샘플이 {output_dir}에 저장되었습니다.")
    
    return points_np, features_np