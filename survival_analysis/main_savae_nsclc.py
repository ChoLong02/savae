# 화순 전대병원 데이터셋으로 중앙화(SAVAE) 실험
#  ㄴ 전처리는 기존 저자랑 동일한 방법

import torch
import numpy as np
import pandas as pd
import random
import sys, os

# 경로 설정 (savae-main 폴더 기준)
sys.path.append('/path/to/savae-main')
sys.path.append('/path/to/savae-main/survival_analysis')

from savae import SAVAE
from data import get_feat_distributions, transform_data, impute_data

# ── 재현성 ─────────────────────────────────────────────────────
# 단일 train/val/test 분할이라 seed에 따라 결과가 흔들립니다.
# 여러 seed로 반복 실행해 평균·표준편차를 함께 보고하시길 권장합니다.
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── 1. 교수님 데이터 로드 ──────────────────────────────────────
# 반드시 'time', 'event' 컬럼 포함, 나머지는 공변량
# event: 1=사건 발생, 0=중도절단
train_df = pd.read_csv('data_preprocessing/raw_data/nsclc/Clinical_Data_Train_3000x10.csv')  # 컬럼: [feat1, feat2, ..., time, event]
valid_df = pd.read_csv('data_preprocessing/raw_data/nsclc/Clinical_Data_Public_504x10.csv')
test_df  = pd.read_csv('data_preprocessing/raw_data/nsclc/Clinical_Data_Test_600x10.csv')

print(f'[LOAD] train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}')

# ── 2. 결측치 처리 및 마스크 생성 ──────────────────────────────
# 결측치가 없으면 모두 1, 있으면 impute_data() 사용
if train_df.isna().any().any() or valid_df.isna().any().any() or test_df.isna().any().any():
    train_df, train_mask = impute_data(train_df)
    valid_df, valid_mask = impute_data(valid_df)
    test_df,  test_mask  = impute_data(test_df)
else:
    train_mask = pd.DataFrame(np.ones(train_df.shape, dtype=int), columns=train_df.columns)
    valid_mask = pd.DataFrame(np.ones(valid_df.shape, dtype=int), columns=valid_df.columns)
    test_mask  = pd.DataFrame(np.ones(test_df.shape,  dtype=int), columns=test_df.columns)

# ── 3. 정규화 (train 통계만으로 fit → valid/test에 apply) ───────
# 엄밀한 "train-only fit" 원칙을 지키기 위해 통계량(mean/std/min)은 train에서만 계산합니다.
# 다만 카테고리 슬롯 수(num_params)만은 concat으로 감지해야 valid/test에 새 카테고리가
# 있을 때도 런타임 에러가 발생하지 않습니다.
#   - 통계 누출 방지: mean/std/min 재계산은 train 전용
#   - 아키텍처 안정성: feat_distributions의 카테고리 범위는 전체 기준
# 로직은 data.py:113-143의 normalize 분기를 그대로 복제한 것입니다.

def compute_norm_params_from_train(train_df, feat_distributions):
    """Train에서만 loc/scale을 계산. 이후 valid/test에 동일 값 적용."""
    params = []
    for i in range(train_df.shape[1]):
        dist = feat_distributions[i][0]
        values = train_df.iloc[:, i]
        no_nan_values = values[~pd.isnull(values)].values
        if dist == 'gaussian':
            loc, scale = float(np.mean(no_nan_values)), float(np.std(no_nan_values))
        elif dist == 'bernoulli':
            loc = float(np.amin(no_nan_values))
            scale = float(np.amax(no_nan_values) - np.amin(no_nan_values))
        elif dist == 'categorical':
            loc, scale = float(np.amin(no_nan_values)), 1.0
        elif dist == 'weibull':
            loc = -1.0 if 0 in no_nan_values else 0.0
            scale = 0.0
        else:
            raise NotImplementedError(f'Distribution {dist} not supported')
        params.append((dist, loc, scale))
    return params


def apply_normalization(df, norm_params):
    """compute_norm_params_from_train()의 결과를 임의의 df에 적용."""
    out = df.copy()
    for i, (dist, loc, scale) in enumerate(norm_params):
        if scale != 0:
            out.iloc[:, i] = (df.iloc[:, i] - loc) / scale
        else:
            out.iloc[:, i] = df.iloc[:, i] - loc
    return out


# (1) 아키텍처 결정용: concat으로 feat_distributions 감지 (카테고리 슬롯 수 보장)
combined_for_schema = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
feat_distributions = get_feat_distributions(combined_for_schema, time=('weibull', 2))

# (2) 통계량 fit: train에서만 mean/std/min 계산
norm_params = compute_norm_params_from_train(train_df, feat_distributions)

# (3) 동일 통계량을 세 세트에 apply
train_df = apply_normalization(train_df, norm_params)
valid_df = apply_normalization(valid_df, norm_params)
test_df  = apply_normalization(test_df,  norm_params)

print(f'[NORM] train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}')
print('[NORM] train head:')
print(train_df.head())
print('[NORM] feat_distributions:', feat_distributions)
print('[NORM] norm_params (from train only):')
for i, (dist, loc, scale) in enumerate(norm_params):
    print(f'  col {i}: {dist:12s} loc={loc:.4f}  scale={scale:.4f}')

# ── 4. SAVAE 모델 초기화 ──────────────────────────────────────
feat_dist_no_sa = feat_distributions[:-2]  # 'time', 'event' 제외한 공변량만
# max_t는 세 세트 모두의 최대 시간을 반영해야 합니다.
max_t = max(train_df['time'].max(), valid_df['time'].max(), test_df['time'].max())

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

# ── 5. 학습 (validation = valid_df) ────────────────────────────
# fit의 4번째 인자가 학습 중 모니터링·early stopping·best model 선택에 사용됩니다.
# test_df는 여기 넣지 않습니다(넣으면 test 누출).
data = (train_df, train_mask, valid_df, valid_mask)

train_params = {
    'n_epochs':   3000,
    'batch_size': 64,
    'lr':         1e-3,
    'device':     torch.device('cpu'),  # GPU: torch.device('cuda')
}
results = model.fit(data, train_params)

# ── 6. 최종 평가 ───────────────────────────────────────────────
# 학습 종료 시점에 SAVAE.fit()은 val loss 최소 시점의 가중치를 이미 로드해 둔 상태입니다
# (savae.py:227). 따라서 아래 calculate_risk 호출은 best model 기준 평가입니다.

# (1) Validation 최종 성능 — 학습 중 참고용 지표
ci_val, ibs_val = model.calculate_risk(
    time_train=np.array(train_df['time']),
    x_val=valid_df,
    censor_val=np.array(valid_df['event'])
)

# (2) Test 최종 성능 — 학습에 전혀 개입하지 않은 독립 평가 (논문·보고용)
ci_test, ibs_test = model.calculate_risk(
    time_train=np.array(train_df['time']),
    x_val=test_df,
    censor_val=np.array(test_df['event'])
)

print('\n===== FINAL METRICS =====')
print(f'[VALID] C-index: {ci_val[1]:.4f}  ({ci_val[0]:.4f} ~ {ci_val[2]:.4f})')
print(f'[VALID] IBS:     {ibs_val[1]:.4f}  ({ibs_val[0]:.4f} ~ {ibs_val[2]:.4f})')
print(f'[TEST ] C-index: {ci_test[1]:.4f}  ({ci_test[0]:.4f} ~ {ci_test[2]:.4f})')
print(f'[TEST ] IBS:     {ibs_test[1]:.4f}  ({ibs_test[0]:.4f} ~ {ibs_test[2]:.4f})')

# 모델 저장
model.save('savae_trained.pt')
print('[SAVE] model → savae_trained.pt')
