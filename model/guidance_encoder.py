import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import PointNetBlock

class GuidanceEncoder(nn.Module):
    """
    이전 프레임의 포인트 클라우드를 인코딩하여 다음 프레임 생성을 위한 가이던스를 제공하는 인코더
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 모델 구성 매개변수
        in_channels = config.model.input_channels  # 좌표 차원 (x,y,z)
        hidden_dim = config.model.hidden_dim
        latent_dim = config.model.get('latent_dim', hidden_dim)  # 잠재 표현 차원
        
        # 인코더 네트워크
        self.encoder_blocks = nn.ModuleList([
            PointNetBlock(in_channels, hidden_dim//4),
            PointNetBlock(hidden_dim//4, hidden_dim//2),
            PointNetBlock(hidden_dim//2, hidden_dim)
        ])
        
        # 전역 특성 추출
        self.global_feat = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 잠재 표현으로 매핑
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 프레임 임베딩 (프레임 번호를 임베딩)
        self.frame_embedding = nn.Embedding(config.data.get('max_frames', 200), latent_dim//4)
        
        # 가이던스 정보 통합 및 처리
        self.guidance_processor = nn.Sequential(
            nn.Linear(latent_dim + latent_dim//4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, points, frame_num):
        """
        이전 프레임의 포인트 클라우드를 인코딩하여 가이던스 정보 생성
        
        Args:
            points: (batch_size, 3, num_points) 텐서, 이전 프레임의 포인트 클라우드
            frame_num: (batch_size,) 텐서, 현재 프레임 번호
            
        Returns:
            (batch_size, latent_dim) 텐서, 다음 프레임 생성을 위한 가이던스 정보
        """
        batch_size = points.shape[0]
        
        # 인코더 블록 통과
        x = points
        for block in self.encoder_blocks:
            x = block(x)
        
        # 전역 특성 추출 (맥스 풀링)
        global_feat = torch.max(self.global_feat(x), dim=2)[0]  # (batch_size, hidden_dim)
        
        # 잠재 표현으로 매핑
        latent = self.to_latent(global_feat)  # (batch_size, latent_dim)
        
        # 프레임 번호 임베딩
        frame_emb = self.frame_embedding(frame_num)  # (batch_size, latent_dim//4)
        
        # 잠재 표현과 프레임 임베딩 결합
        combined = torch.cat([latent, frame_emb], dim=1)  # (batch_size, latent_dim + latent_dim//4)
        
        # 가이던스 정보 생성
        guidance = self.guidance_processor(combined)  # (batch_size, latent_dim)
        
        return guidance


class GuidedPointDiffusion(nn.Module):
    """
    이전 프레임의 가이던스를 활용하는 포인트 클라우드 디퓨전 모델
    """
    def __init__(self, diffusion_model, guidance_encoder):
        super().__init__()
        self.diffusion_model = diffusion_model  # 기존 디퓨전 모델
        self.guidance_encoder = guidance_encoder  # 가이던스 인코더
        
        # 가이던스 통합을 위한 추가 레이어
        hidden_dim = diffusion_model.encoder.block3.conv2.out_channels
        latent_dim = guidance_encoder.to_latent[2].out_features
        
        # 디버깅 정보 출력
        print(f"Diffusion model encoder output dimension: {hidden_dim}")
        print(f"Guidance encoder latent dimension: {latent_dim}")
        
        # 특성 차원 추정
        # 인코더 출력은 일반적으로 hidden_dim*2 (로컬 특성 + 전역 특성 concat)
        encoder_out_dim = hidden_dim * 2
        
        # 가이던스 정보 투영 레이어 - 출력 차원을 encoder_out_dim과 맞춤
        self.guidance_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_out_dim)
        )
        
        print(f"Guidance projector output dimension: {encoder_out_dim}")
    
    def forward(self, x, t, prev_points=None, frame_num=None):
        """
        노이즈 예측 (가이던스 정보 활용)
        
        Args:
            x: (batch_size, 3, num_points) 텐서, 노이즈가 추가된 포인트 클라우드
            t: (batch_size,) 텐서, 디퓨전 타임스텝
            prev_points: (batch_size, 3, num_points) 텐서, 이전 프레임 포인트 클라우드 (선택적)
            frame_num: (batch_size,) 텐서, 현재 프레임 번호 (선택적)
            
        Returns:
            (batch_size, 3, num_points) 텐서, 예측된 노이즈
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 가이던스 정보가 제공될 경우에만 가이던스 활용
        if prev_points is not None and frame_num is not None:
            # 가이던스 인코더로 이전 프레임 인코딩
            guidance_info = self.guidance_encoder(prev_points, frame_num)
            
            # 시간 임베딩 생성
            time_emb = self.diffusion_model.time_mlp(t)
            
            # 인코더로 특성 추출
            features = self.diffusion_model.encoder(x, t)
            
            # 디버깅 정보 출력
            if batch_size == 1:  # 배치 크기 1일 때만 출력하여 로그 과부하 방지
                print(f"Features shape: {features.shape}")
            
            # 가이던스 정보 투영
            guidance_feat = self.guidance_projector(guidance_info)  # (batch_size, encoder_out_dim)
            
            # 디버깅 정보 출력
            if batch_size == 1:
                print(f"Guidance feat shape before expansion: {guidance_feat.shape}")
            
            # 가이던스 특성을 모든 포인트에 확장
            guidance_feat = guidance_feat.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            
            # 차원 확인
            if guidance_feat.shape[1] != features.shape[1]:
                raise ValueError(f"Feature dimensions mismatch: features={features.shape}, guidance_feat={guidance_feat.shape}")
            
            # 특성과 가이던스 정보 결합
            enhanced_features = features + guidance_feat  # 가이던스 정보 추가
            
            # 디코더로 노이즈 예측
            noise_pred = self.diffusion_model.decoder(enhanced_features, time_emb)
        else:
            # 가이던스 없이 기본 디퓨전 모델 사용
            noise_pred = self.diffusion_model(x, t)
        
        return noise_pred


def create_guided_diffusion_model(config):
    """
    가이던스 인코더가 통합된 디퓨전 모델 생성
    
    Args:
        config: 모델 설정
        
    Returns:
        GuidedPointDiffusion: 가이던스 기능이 있는 디퓨전 모델
    """
    from .diffusion import PointDiffusionModel
    
    # 기본 디퓨전 모델 생성
    diffusion_model = PointDiffusionModel(config)
    
    # 가이던스 인코더 생성
    guidance_encoder = GuidanceEncoder(config)
    
    # 가이던스 통합 모델 생성
    guided_model = GuidedPointDiffusion(diffusion_model, guidance_encoder)
    
    # 모델을 적절한 디바이스로 이동
    device = torch.device(config.system.device)
    guided_model = guided_model.to(device)
    
    # 다중 GPU 사용 설정
    if torch.cuda.device_count() > 1 and config.system.get('multi_gpu', False):
        print(f"다중 GPU 학습 활성화: {torch.cuda.device_count()}개 GPU 사용")
        guided_model = nn.DataParallel(guided_model)
    
    return guided_model