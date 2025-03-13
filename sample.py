import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.config import get_config
from model import GuidedPointDiffusion, create_guided_diffusion_model
from model.diffusion import PointCloudDiffusion

def generate_samples(config, model_path, num_samples=10, save_dir="result"):
    """
    학습된 모델로 샘플 생성
    
    Args:
        config: 설정 객체
        model_path: 학습된 모델 경로
        num_samples: 생성할 샘플 수
        save_dir: 결과 저장 경로
    """
    device = torch.device(config.system.device)
    
    # 데이터로더에서 평균 포인트 수 가져오기 (config에 설정이 없을 경우)
    if config.sampling.num_points <= 0:
        from datasets.frame_dataloader import get_dataloader
        _, avg_points = get_dataloader(config.data)
        num_points = avg_points
        print(f"샘플링 포인트 수를 데이터셋 평균으로 설정: {num_points}")
    else:
        num_points = config.sampling.num_points
    
    # 디퓨전 모델 초기화
    diffusion = PointCloudDiffusion(config)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    diffusion.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 샘플 생성
    use_ddim = config.sampling.get('use_ddim', True)
    ddim_steps = config.sampling.get('ddim_steps', 50)
    ddim_eta = config.sampling.get('ddim_eta', 0.0)
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # 샘플 생성
        if use_ddim:
            # DDIM 샘플링 (더 빠르고 품질이 좋음)
            points = diffusion.p_sample_loop_ddim(
                shape=(1, 3, num_points),
                ddim_steps=ddim_steps,
                eta=ddim_eta
            )
        else:
            # 기본 DDPM 샘플링
            points = diffusion.p_sample_loop(
                shape=(1, 3, num_points)
            )
        
        # 텐서를 NumPy로 변환
        points = points.transpose(1, 2).cpu().numpy()[0]  # [num_points, 3]
        
        # CSV로 저장
        df = pd.DataFrame(points, columns=['x', 'y', 'z'])
        
        # 빈 특성 값 추가 (원본 데이터 형식 유지)
        df['scl1'] = 0.0
        
        # 파일 저장
        file_path = os.path.join(save_dir, f"generated_point_cloud_{i+1}.csv")
        df.to_csv(file_path, index=False)
        print(f"Sample saved to {file_path}")


def generate_frame_sequence(config, model_path, start_frame=100, end_frame=200, save_dir="result"):
    """
    연속적인 프레임 시퀀스 생성
    
    Args:
        config: 설정 객체
        model_path: 학습된 모델 경로
        start_frame: 시작 프레임 번호
        end_frame: 종료 프레임 번호
        save_dir: 결과 저장 경로
    """
    device = torch.device(config.system.device)
    
    # 가이던스 기반 디퓨전 모델 초기화
    diffusion_model = create_guided_diffusion_model(config)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()
    print(f"Model loaded from {model_path}")
    
    # 디퓨전 프로세스 초기화
    diffusion = PointCloudDiffusion(config)
    diffusion.model = diffusion_model
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 샘플링 설정
    num_points = config.sampling.num_points
    ddim_steps = config.sampling.get('ddim_steps', 50)
    ddim_eta = config.sampling.get('ddim_eta', 0.0)
    
    # 첫 프레임 생성 (가이던스 없이)
    print(f"Generating first frame ({start_frame})...")
    first_frame = diffusion.p_sample_loop_ddim(
        shape=(1, 3, num_points),
        ddim_steps=ddim_steps,
        eta=ddim_eta
    )
    
    # 첫 프레임 저장
    first_frame_points = first_frame.transpose(1, 2).cpu().numpy()[0]  # [N, 3]
    first_frame_file = os.path.join(save_dir, f"frame_{start_frame}.csv")
    df = pd.DataFrame(first_frame_points, columns=['x', 'y', 'z'])
    df['scl1'] = 0.0  # 더미 특성값 추가
    df.to_csv(first_frame_file, index=False)
    print(f"First frame saved to {first_frame_file}")
    
    # 이전 프레임을 사용하여 연속적으로 다음 프레임 생성
    prev_frame = first_frame
    
    for frame_num in tqdm(range(start_frame + 1, end_frame + 1), desc="Generating frames"):
        # 현재 프레임 번호 텐서 생성
        current_frame_tensor = torch.tensor([frame_num], device=device)
        
        # 노이즈에서 시작
        x_t = torch.randn((1, 3, num_points), device=device)
        
        # DDIM 샘플링 스텝
        for i in tqdm(range(len(diffusion.alphas_cumprod) - 1, 0, -diffusion.num_steps // ddim_steps), 
                     desc=f"Sampling frame {frame_num}", leave=False):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            
            # 가이던스 정보 사용하여 노이즈 예측
            with torch.no_grad():
                noise_pred = diffusion_model(x_t, t, prev_frame, current_frame_tensor)
            
            # 다음 샘플링 스텝 계산
            alpha_cumprod_t = diffusion.alphas_cumprod[i]
            alpha_cumprod_t_prev = diffusion.alphas_cumprod[i-diffusion.num_steps // ddim_steps] if i >= diffusion.num_steps // ddim_steps else torch.tensor(1.0).to(device)
            
            # x_0 예측
            x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # 방향 벡터 계산
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - ddim_eta**2 * (1 - alpha_cumprod_t_prev) * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t_prev)) * noise_pred
            
            # 노이즈 생성 (eta > 0 일 경우)
            if ddim_eta > 0:
                noise = torch.randn_like(x_t)
                dir_xt += ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t_prev)) * noise
            
            # 다음 샘플 계산
            x_t = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt
        
        # 현재 생성된 프레임 저장
        current_frame_points = x_t.transpose(1, 2).cpu().numpy()[0]  # [N, 3]
        current_frame_file = os.path.join(save_dir, f"frame_{frame_num}.csv")
        df = pd.DataFrame(current_frame_points, columns=['x', 'y', 'z'])
        df['scl1'] = 0.0  # 더미 특성값 추가
        df.to_csv(current_frame_file, index=False)
        print(f"Frame {frame_num} saved to {current_frame_file}")
        
        # 현재 프레임을 다음 프레임의 가이던스로 사용
        prev_frame = x_t
    
    print(f"Generated {end_frame - start_frame + 1} frames in sequence at {save_dir}")


def main():
    """
    메인 샘플링 함수
    """
    config = get_config()
    
    # 모델 경로
    model_path = config.get('model_path', os.path.join(config.system.save_dir, "model_final.pt"))
    
    # 결과 저장 경로
    save_dir = config.sampling.save_dir
    
    # 항상 프레임 시퀀스 생성 방식 사용
    generate_frame_sequence(
        config=config,
        model_path=model_path,
        start_frame=config.data.get('min_frame', 100),
        end_frame=config.data.get('max_frame', 200),
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()