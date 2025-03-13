import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class SinusoidalPositionEmbeddings(nn.Module):
    """
    시간 스텝을 인코딩하기 위한 사인 위치 임베딩
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch_size,) 텐서, 시간 스텝
        Returns:
            (batch_size, dim) 텐서, 시간 임베딩
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PointNetBlock(nn.Module):
    """
    PointNet 스타일의 포인트 클라우드 처리 블록
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 스킵 커넥션을 위한 1x1 컨볼루션
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
        # 시간 임베딩 사용 시 추가 레이어
        self.has_time_emb = time_emb_dim is not None
        if self.has_time_emb:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, time_emb=None):
        """
        Args:
            x: (batch_size, in_channels, num_points) 텐서
            time_emb: (batch_size, time_emb_dim) 텐서, 선택적
        Returns:
            (batch_size, out_channels, num_points) 텐서
        """
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        
        # 시간 임베딩 주입
        if self.has_time_emb and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb[..., None].expand(-1, -1, x.shape[-1])
            x = x + time_emb
            
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # 스킵 커넥션 적용
        if self.shortcut is not None:
            identity = self.shortcut(identity)
            
        x = F.relu(x + identity)
        
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet 기반 인코더
    """
    def __init__(self, in_channels, hidden_dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.block1 = PointNetBlock(in_channels, hidden_dim//4, time_emb_dim)
        self.block2 = PointNetBlock(hidden_dim//4, hidden_dim//2, time_emb_dim)
        self.block3 = PointNetBlock(hidden_dim//2, hidden_dim, time_emb_dim)
        
        # 전역 특성 처리
        self.global_feat = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, time):
        """
        Args:
            x: (batch_size, in_channels, num_points) 텐서
            time: (batch_size,) 텐서, 시간 스텝
        Returns:
            (batch_size, hidden_dim, num_points) 텐서
        """
        time_emb = self.time_mlp(time)
        
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        x = self.block3(x, time_emb)
        
        # 맥스 풀링으로 전역 특성 추출
        global_feat = torch.max(self.global_feat(x), dim=2, keepdim=True)[0]
        # 전역 특성을 모든 포인트에 확장
        global_feat = global_feat.expand(-1, -1, x.shape[2])
        
        # 전역 특성과 로컬 특성 결합
        x = torch.cat([x, global_feat], dim=1)
        
        return x


class PointNetDecoder(nn.Module):
    """
    PointNet 기반 디코더
    """
    def __init__(self, in_channels, hidden_dim, out_channels, time_emb_dim):
        super().__init__()
        self.block1 = PointNetBlock(in_channels, hidden_dim, time_emb_dim)
        self.block2 = PointNetBlock(hidden_dim, hidden_dim//2, time_emb_dim)
        self.block3 = PointNetBlock(hidden_dim//2, hidden_dim//4, time_emb_dim)
        
        self.final_conv = nn.Conv1d(hidden_dim//4, out_channels, 1)

    def forward(self, x, time_emb):
        """
        Args:
            x: (batch_size, in_channels, num_points) 텐서
            time_emb: (batch_size, time_emb_dim) 텐서
        Returns:
            (batch_size, out_channels, num_points) 텐서
        """
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        x = self.block3(x, time_emb)
        
        x = self.final_conv(x)
        
        return x


class PointDiffusionModel(nn.Module):
    """
    포인트 클라우드 디퓨전 모델
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 장치 설정
        self.device = torch.device(config.system.device)
        
        # 모델 구성 매개변수
        in_channels = config.model.input_channels  # 좌표 차원 (x,y,z)
        hidden_dim = config.model.hidden_dim
        time_emb_dim = hidden_dim
        
        # 시간 임베딩
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        ).to(self.device)
        
        # 인코더-디코더 구조
        self.encoder = PointNetEncoder(in_channels, hidden_dim, time_emb_dim).to(self.device)
        self.decoder = PointNetDecoder(hidden_dim*2, hidden_dim, in_channels, time_emb_dim).to(self.device)
        
        # # 다중 GPU 사용을 위한 설정
        # if torch.cuda.device_count() > 1 and config.system.get('multi_gpu', False):
        #     print(f"Using {torch.cuda.device_count()} GPUs for parallel training")
        #     self.is_parallel = True
        #     self.encoder = nn.DataParallel(self.encoder)
        #     self.decoder = nn.DataParallel(self.decoder)
        #     self.time_mlp = nn.DataParallel(self.time_mlp)
        # else:
        #     self.is_parallel = False
    
    def forward(self, x, t):
        """
        Args:
            x: (batch_size, 3, num_points) 텐서, 노이즈가 추가된 포인트 클라우드
            t: (batch_size,) 텐서, 디퓨전 타임스텝
        Returns:
            (batch_size, 3, num_points) 텐서, 예측된 노이즈
        """
        time_emb = self.time_mlp(t)
        
        # 인코더로 특성 추출
        features = self.encoder(x, t)
        
        # 디코더로 노이즈 예측
        noise_pred = self.decoder(features, time_emb)
        
        return noise_pred


class PointCloudDiffusion:
    """
    포인트 클라우드 디퓨전 프로세스 관리 클래스
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # 디퓨전 파라미터
        self.num_steps = config.model.num_steps
        self.beta_start = config.model.beta_start
        self.beta_end = config.model.beta_end
        
        # 베타 스케줄 설정
        self.beta_schedule = config.model.beta_schedule
        
        # 베타 스케줄 초기화
        self._init_beta_schedule()
        
        # 모델 초기화 및 디바이스로 이동
        from .guidance_encoder import create_guided_diffusion_model
        self.model = create_guided_diffusion_model(config)
        self.model = self.model.to(self.device)
        
        # 다중 GPU 적용 (모델이 디바이스로 이동한 후)
        if torch.cuda.device_count() > 1 and config.system.get('multi_gpu', False):
            print(f"다중 GPU 학습 활성화: {torch.cuda.device_count()}개 GPU 사용")
            self.model = nn.DataParallel(self.model)
    
    def _init_beta_schedule(self):
        """
        베타 스케줄 초기화
        """
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps).to(self.device)
        elif self.beta_schedule == "cosine":
            # 코사인 스케줄링
            steps = self.num_steps + 1
            x = torch.linspace(0, self.num_steps, steps)
            alphas_cumprod = torch.cos(((x / self.num_steps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999).to(self.device)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # 알파 값 계산
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 샘플링에 필요한 계수 계산
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 사후 분산 계산
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: (batch_size, 3, num_points) 텐서, 원본 포인트 클라우드
            t: (batch_size,) 텐서, 시간 스텝
            noise: (batch_size, 3, num_points) 텐서, 노이즈 (없으면 생성)
            
        Returns:
            (batch_size, 3, num_points) 텐서, t 시간에서의 노이즈가 추가된 포인트 클라우드
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, t, noise=None):
        """
        학습 손실 계산
        
        Args:
            x_0: (batch_size, 3, num_points) 텐서, 원본 포인트 클라우드
            t: (batch_size,) 텐서, 시간 스텝
            noise: (batch_size, 3, num_points) 텐서, 노이즈 (없으면 생성)
            
        Returns:
            loss: 스칼라 텐서, 노이즈 예측 손실
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # t 시간의 노이즈 추가된 샘플 생성
        x_t = self.q_sample(x_0, t, noise)
        
        # 모델로 노이즈 예측
        noise_pred = self.model(x_t, t)
        
        # L2 손실 계산
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def p_sample(self, x_t, t):
        """
        단일 역 확산 스텝: p(x_{t-1} | x_t)
        DDPM 샘플링 알고리즘
        
        Args:
            x_t: (batch_size, 3, num_points) 텐서, t 시간의 노이즈화된 포인트 클라우드
            t: 스칼라 텐서, 현재 시간 스텝
            
        Returns:
            (batch_size, 3, num_points) 텐서, 하나의 스텝을 역확산한 결과
        """
        self.model.eval()
        with torch.no_grad():
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            # 노이즈 예측
            noise_pred = self.model(x_t, t_tensor)
            
            # 파라미터 계산
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            beta = self.betas[t]
            
            # 사후 분포 평균 계산
            coef1 = beta * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            coef2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
            posterior_mean = coef1 * x_t + coef2 * (1.0 / torch.sqrt(alpha)) * (x_t - noise_pred * torch.sqrt(1 - alpha_cumprod))
            
            # 사후 분포 분산 계산
            posterior_variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            
            # 노이즈 추가 (t > 0 일 때만)
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t_1 = posterior_mean + torch.sqrt(posterior_variance) * noise
            
            return x_t_1
    
    def p_sample_loop(self, shape):
        """
        전체 역 확산 프로세스
        
        Args:
            shape: 생성할 포인트 클라우드의 형태 (batch_size, 3, num_points)
            
        Returns:
            (batch_size, 3, num_points) 텐서, 생성된 포인트 클라우드
        """
        self.model.eval()
        device = self.device
        
        batch_size = shape[0]
        # 무작위 노이즈로 시작
        x_t = torch.randn(shape, device=device)
        
        # 역 확산 스텝
        for t in tqdm(reversed(range(self.num_steps)), desc="Sampling", total=self.num_steps):
            x_t = self.p_sample(x_t, t)
            
        return x_t
    
    def p_sample_ddim(self, x_t, t, t_prev, eta=0.0):
        """
        DDIM 샘플링 스텝
        
        Args:
            x_t: (batch_size, 3, num_points) 텐서, t 시간의 노이즈화된 포인트 클라우드
            t: 스칼라 텐서, 현재 시간 스텝
            t_prev: 스칼라 텐서, 이전 시간 스텝
            eta: 스칼라, 노이즈 스케일 (0: 결정적, 1: 확률적)
            
        Returns:
            (batch_size, 3, num_points) 텐서, DDIM으로 샘플링된 결과
        """
        self.model.eval()
        with torch.no_grad():
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            # 노이즈 예측
            noise_pred = self.model(x_t, t_tensor)
            
            # 알파 값 로드
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
            
            # x_0 예측
            x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # 방향 벡터 계산
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev) * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t_prev)) * noise_pred
            
            # 노이즈 생성 (eta > 0 일 경우)
            if eta > 0:
                noise = torch.randn_like(x_t)
                dir_xt += eta * torch.sqrt((1 - alpha_cumprod_t_prev) * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t_prev)) * noise
            
            # 다음 샘플 계산
            x_t_prev = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt
            
            return x_t_prev
    
    def p_sample_loop_ddim(self, shape, ddim_steps=50, eta=0.0):
        """
        DDIM 샘플링 프로세스
        
        Args:
            shape: 생성할 포인트 클라우드의 형태
            ddim_steps: DDIM 스텝 수
            eta: 노이즈 스케일 (0: 결정적, 1: 확률적)
            
        Returns:
            (batch_size, 3, num_points) 텐서, 생성된 포인트 클라우드
        """
        self.model.eval()
        device = self.device
        
        # 무작위 노이즈로 시작
        x_t = torch.randn(shape, device=device)
        
        # DDIM 시간 스텝 계산
        times = torch.linspace(self.num_steps-1, 0, ddim_steps+1).long().to(device)
        
        # 역 확산 스텝
        for i in tqdm(range(len(times)-1), desc="DDIM Sampling", total=len(times)-1):
            t = times[i]
            t_prev = times[i+1]
            x_t = self.p_sample_ddim(x_t, t, t_prev, eta)
            
        return x_t
    
    def train(self, dataloader, optimizer, epochs, save_dir, log_interval=10, save_interval=10):
        """
        모델 학습 (가이던스 기반)
        
        Args:
            dataloader: 학습 데이터로더
            opt imizer: 최적화 알고리즘
            epochs: 학습 에폭 수
            save_dir: 모델 저장 경로
            log_interval: 로그 출력 간격 (배치)
            save_interval: 모델 저장 간격 (에폭)
        """
        device = self.device
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=0, leave=True)
            
            for batch_idx, batch in enumerate(progress_bar):
                # 현재 및 다음 프레임 데이터 로드
                current_points = batch['current_points'].to(device)
                current_points = current_points.transpose(1, 2)  # [B, N, 3] -> [B, 3, N]
                next_points = batch['next_points'].to(device)
                next_points = next_points.transpose(1, 2)  # [B, N, 3] -> [B, 3, N]
                current_frame = batch['current_frame'].to(device)
                next_frame = batch['next_frame'].to(device)
                
                # 옵티마이저 초기화
                optimizer.zero_grad()
                
                # 랜덤 타임스텝 선택
                batch_size = next_points.shape[0]
                t = torch.randint(0, self.num_steps, (batch_size,), device=device).long()
                
                # 가이던스 기반 손실 계산
                # 다음 프레임에 노이즈 추가
                noise = torch.randn_like(next_points)
                x_t = self.q_sample(next_points, t, noise)
                
                # 현재 프레임 정보를 사용한 노이즈 예측
                noise_pred = self.model(x_t, t, current_points, next_frame)
                
                # 손실 계산
                loss = F.mse_loss(noise_pred, noise)
                
                # 역전파
                loss.backward()
                
                # 그래디언트 클리핑 (필요시)
                if self.config.train.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip)
                
                # 파라미터 업데이트
                optimizer.step()
                
                # 손실 누적 및 진행 상태 업데이트
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.6f}")
                
                # 로그 출력
                if batch_idx % log_interval == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
            
            # 에폭 완료 후 평균 손실 출력
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_epoch_loss:.6f}")
            
            # 모델 저장
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
                print(f"Model saved at epoch {epoch+1}")
                
                # 샘플 생성 및 저장 (학습 진행 상황 모니터링용)
                if self.config.sampling.save_dir is not None:
                    os.makedirs(self.config.sampling.save_dir, exist_ok=True)
                    with torch.no_grad():
                        # 빠른 평가를 위해 DDIM 샘플링 사용
                        sample_points = self.p_sample_loop_ddim(
                            shape=(1, 3, self.config.sampling.num_points),
                            ddim_steps=self.config.sampling.ddim_steps,
                            eta=self.config.sampling.ddim_eta
                        )
                        
                        # 샘플 저장
                        sample_points = sample_points.transpose(1, 2).cpu().numpy()[0]  # [N, 3]
                        sample_file = os.path.join(self.config.sampling.save_dir, f"sample_epoch_{epoch+1}.csv")
                        np.savetxt(sample_file, sample_points, delimiter=',', header='x,y,z', comments='')
                        print(f"Sample saved at {sample_file}")
                
        return self.model