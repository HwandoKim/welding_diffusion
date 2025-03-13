import os
import yaml
from collections import defaultdict

class Config(dict):
    """
    EasyDict 대신 사용할 수 있는 중첩 딕셔너리 클래스
    """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        return None

def get_config(config_file=None):
    """
    Config 설정을 로드하는 함수
    
    Args:
        config_file (str, optional): 설정 파일 경로
        
    Returns:
        Config: 설정 정보가 담긴 사전
    """
    # 기본 설정
    config = Config()
    
    # 데이터 관련 설정
    config.data = Config()
    config.data.data_dir = "/workspace/welding_paper_project/data/bead_all_data"                # 데이터 디렉토리 경로
    config.data.num_workers = 4                  # 데이터 로딩에 사용할 워커 수
    config.data.pin_memory = True                # GPU 사용 시 메모리 핀 설정
    config.data.shuffle = True                   # 데이터 셔플 여부
    config.data.normalize = True                 # 포인트 클라우드 정규화 여부
    
    # 모델 관련 설정
    config.model = Config()
    config.model.type = "diffusion"              # 모델 타입
    config.model.input_channels = 3              # 입력 채널 수 (x, y, z 좌표)
    config.model.feature_channels = 2            # 특성 채널 수 (f, scl1)
    config.model.hidden_dim = 512                # 히든 레이어 차원
    config.model.num_steps = 1000                # 디퓨전 스텝 수
    config.model.beta_schedule = "linear"        # 베타 스케줄 (linear, cosine)
    config.model.beta_start = 1e-4               # 베타 시작값
    config.model.beta_end = 2e-2                 # 베타 종료값
    
    # 학습 관련 설정
    config.train = Config()
    config.train.batch_size = 1                  # 배치 사이즈 (고정)
    config.train.lr = 1e-4                       # 학습률
    config.train.weight_decay = 1e-5             # 가중치 감쇠
    config.train.epochs = 100                    # 총 에폭 수
    config.train.save_interval = 10              # 모델 저장 간격 (에폭)
    config.train.log_interval = 10               # 로그 출력 간격 (배치)
    config.train.optimizer = "adam"              # 옵티마이저 (adam, adamw)
    config.train.scheduler = "cosine"            # 스케줄러 (step, cosine)
    config.train.gradient_clip = 1.0             # 그래디언트 클리핑 값
    
    # 샘플링 관련 설정
    config.sampling = Config()
    config.sampling.num_samples = 10             # 생성할 샘플 수
    config.sampling.save_dir = "result/guidance_model"          # 생성 결과 저장 경로
    config.sampling.num_points = -1              # 생성할 포인트 수 (-1: 데이터셋 평균 포인트 수 사용)
    config.sampling.ddim_steps = 100              # DDIM 샘플링 스텝 수 (가속 샘플링)
    config.sampling.ddim_eta = 0.0               # DDIM 노이즈 스케일 (0: 결정적, 1: 확률적)
    
    # 시스템 관련 설정
    config.system = Config()
    config.system.seed = 42                      # 랜덤 시드
    config.system.save_dir = "saved"             # 모델 저장 경로
    config.system.device = "cuda"                # 사용할 디바이스 (cuda, cpu)
    config.system.precision = "fp32"             # 연산 정밀도 (fp32, fp16)
    config.system.multi_gpu = True               # 다중 GPU 병렬 학습 사용
    config.system.distributed = False            # 분산 학습 사용 (DistributedDataParallel)
    # config/config.py 파일에 추가할 내용

    # 데이터 관련 설정에 프레임 정보 추가
    config.data.min_frame = 100           # 시작 프레임 번호
    config.data.max_frames = 200          # 최대 프레임 번호
    config.data.frame_based = True        # 프레임 기반 데이터셋 사용 여부

    # 모델 관련 설정에 가이던스 관련 파라미터 추가
    config.model.use_guidance = True      # 가이던스 인코더 사용 여부
    config.model.latent_dim = 256         # 가이던스 잠재 표현 차원
    config.model.guidance_strength = 1.0  # 가이던스 영향력 가중치
    # 설정 파일이 제공된 경우 해당 설정 로드
    if config_file is not None and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
            _update_config(config, loaded_config)
    
    return config

def _update_config(config, update_dict):
    """
    설정 사전을 업데이트하는 함수
    
    Args:
        config (Config): 업데이트할 설정
        update_dict (dict): 업데이트 내용
    """
    for k, v in update_dict.items():
        if k in config and isinstance(v, dict):
            _update_config(config[k], v)
        else:
            config[k] = v

def save_config(config, config_file):
    """
    설정을 파일로 저장하는 함수
    
    Args:
        config (Config): 저장할 설정
        config_file (str): 저장할 파일 경로
    """
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # 설정을 일반 딕셔너리로 변환
    def convert_to_dict(cfg):
        if isinstance(cfg, Config):
            return {k: convert_to_dict(v) for k, v in cfg.items()}
        return cfg
    
    config_dict = convert_to_dict(config)
    
    # YAML 파일로 저장
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

if __name__ == "__main__":
    # 기본 설정 생성 및 출력
    config = get_config()
    print(config)
    
    # 설정 파일 저장 예시
    save_config(config, "config/default_config.yaml")