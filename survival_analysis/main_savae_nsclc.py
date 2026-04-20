import torch
import numpy as np
import pandas as pd
import sys, os

# 경로 설정 (savae-main 폴더 기준)
sys.path.append('/path/to/savae-main')
sys.path.append('/path/to/savae-main/survival_analysis')

from savae import SAVAE
from data import get_feat_distributions, transform_data, impute_data

# ── 1. 교수님 데이터 로드 ──────────────────────────────────────
# 반드시 'time', 'event' 컬럼 포함, 나머지는 공변량
# event: 1=사건 발생, 0=중도절단
train_df = pd.read_csv('your_train.csv')  # 컬럼: [feat1, feat2, ..., time, event]
test_df  = pd.read_csv('your_test.csv')

# ── 2. 결측치 마스크 생성 ──────────────────────────────────────
# 결측치가 없으면 모두 1, 있으면 impute_data() 사용
if train_df.isna().any().any() or test_df.isna().any().any():
    train_df, train_mask = impute_data(train_df)
    test_df,  test_mask  = impute_data(test_df)
else:
    train_mask = pd.DataFrame(np.ones_like(train_df), columns=train_df.columns).astype(int)
    test_mask  = pd.DataFrame(np.ones_like(test_df),  columns=test_df.columns).astype(int)

# ── 3. 정규화 ─────────────────────────────────────────────────
# 이미 정규화가 완료된 경우 이 단계 생략 가능
# (단, 분포 추론은 정규화 전 원본 데이터로 수행 권장)
feat_distributions = get_feat_distributions(train_df, time=('weibull', 2))
train_df = transform_data(train_df, feat_distributions)
test_df  = transform_data(test_df,  feat_distributions)

# ── 4. SAVAE 모델 초기화 ──────────────────────────────────────
feat_dist_no_sa = feat_distributions[:-2]  # 'time', 'event' 제외한 공변량만
max_t = max(train_df['time'].max(), test_df['time'].max())

model_params = {
    'feat_distributions': feat_dist_no_sa,
    'input_dim':   len(feat_dist_no_sa),
    'latent_dim':  5,       # 하이퍼파라미터
    'hidden_size': 50,      # 하이퍼파라미터
    'dropout_prop': 0.2,
    'max_t':       max_t,
    'time_dist':   ('weibull', 2),
    'early_stop':  True,
}
model = SAVAE(model_params)

# ── 5. 학습 ───────────────────────────────────────────────────
data = (train_df, train_mask, test_df, test_mask)

train_params = {
    'n_epochs':   3000,
    'batch_size': 64,
    'lr':         1e-3,
    'device':     torch.device('cpu'),  # GPU: torch.device('cuda')
}
results = model.fit(data, train_params)

# ── 6. 예측 및 평가 ───────────────────────────────────────────
ci, ibs = model.calculate_risk(
    time_train=np.array(train_df['time']),
    x_val=test_df,
    censor_val=np.array(test_df['event'])
)
print(f'C-index: {ci[1]:.4f}  ({ci[0]:.4f} ~ {ci[2]:.4f})')
print(f'IBS:     {ibs[1]:.4f}  ({ibs[0]:.4f} ~ {ibs[2]:.4f})')

# 모델 저장
model.save('savae_trained.pt')