import os
import torch
import numpy as np
import random
from tqdm import tqdm
from config.diffusion_config import get_config, save_config
from datasets.frame_dataloader import get_dataloader
from model import PointCloudDiffusion
from model.guidance_encoder import create_guided_diffusion_model
import torch.nn as nn

def set_seed(seed):
    """
    랜덤 시드 설정
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """
    메인 학습 함수
    """
    # 설정 로드
    config = get_config()
    
    # 시드 설정
    set_seed(config.system.seed)
    
    # 장치 설정
    device = torch.device(config.system.device)
    print(f"Using device: {device}")
    
    # 다중 GPU 설정 확인
    if torch.cuda.device_count() > 1 and config.system.multi_gpu:
        print(f"다중 GPU 학습 활성화: {torch.cuda.device_count()}개 GPU 사용")
        if config.system.distributed:
            print("분산 학습 모드 활성화 (DistributedDataParallel)")
        else:
            print("데이터 병렬 모드 활성화 (DataParallel)")
    
    # 프레임 기반 데이터로더 생성
    print("프레임 기반 데이터로더 생성...")
    dataloader, avg_points = get_dataloader(config.data)
    
    # 평균 포인트 수를 config에 저장
    if config.sampling.num_points == -1:
        config.sampling.num_points = avg_points
        print(f"샘플링 포인트 수를 데이터셋 평균으로 설정: {avg_points}")
    
    # 가이던스 인코더를 사용하는 디퓨전 모델 초기화
    print("가이던스 인코더를 사용하는 디퓨전 모델 초기화...")
    diffusion = PointCloudDiffusion(config)
    
    
    # 최적화기 설정
    if config.train.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            diffusion.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            diffusion.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer}")
    
    # 학습률 스케줄러 설정
    if config.train.scheduler.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.epochs // 3,
            gamma=0.5
        )
    elif config.train.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.lr * 0.01
        )
    else:
        scheduler = None
    
    # 학습 실행
    print(f"Starting training for {config.train.epochs} epochs...")
    os.makedirs(config.system.save_dir, exist_ok=True)
    
    # 현재 설정 저장
    save_config(config, os.path.join(config.system.save_dir, 'config.yaml'))
    
    # 학습 시작
    diffusion.train(
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=config.train.epochs,
        save_dir=config.system.save_dir,
        log_interval=config.train.log_interval,
        save_interval=config.train.save_interval
    )
    
    # 스케줄러 적용
    if scheduler is not None:
        scheduler.step()
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': diffusion.model.state_dict(),
    }, os.path.join(config.system.save_dir, "model_final.pt"))
    print("Training completed!")

if __name__ == "__main__":
    main()