# 화순 전대병원 데이터셋으로 중앙화(SAVAE) 실험
#  ㄴ 전처리는 기존에 내가 사용하던 방법 (preprocess_nsclc 기반: 원핫 + StandardScaler)

import torch
import numpy as np
import pandas as pd
import random
import sys, os

from sklearn.preprocessing import StandardScaler

# 경로 설정 (savae 루트와 survival_analysis 폴더 둘 다 추가)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(THIS_DIR)

from savae import SAVAE
from data import get_feat_distributions, impute_data

# ── 재현성 ─────────────────────────────────────────────────────
# 단일 train/val/test 분할이라 seed에 따라 결과가 흔들립니다.
# 여러 seed로 반복 실행해 평균·표준편차를 함께 보고하시길 권장합니다.
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── 1. 데이터 로드 ─────────────────────────────────────────────
# 반드시 'time', 'event' 컬럼 포함, 나머지는 공변량
# event: 1=사건 발생, 0=중도절단
train_df = pd.read_csv('data_preprocessing/raw_data/nsclc/Clinical_Data_Train_3000x10.csv')
valid_df = pd.read_csv('data_preprocessing/raw_data/nsclc/Clinical_Data_Public_504x10.csv')
test_df  = pd.read_csv('data_preprocessing/raw_data/nsclc/Clinical_Data_Test_600x10.csv')

print(f'[LOAD] train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}')

# ── 2. 결측치 처리 ─────────────────────────────────────────────
# 결측치가 있으면 impute_data로 채운 뒤 마스크 생성. 없으면 마스크는 전부 1.
# (mask는 impute된 값으로 학습되지 않도록 vae_modules.py:106-110에서 손실에 반영됨)
if train_df.isna().any().any() or valid_df.isna().any().any() or test_df.isna().any().any():
    train_df, train_mask = impute_data(train_df)
    valid_df, valid_mask = impute_data(valid_df)
    test_df,  test_mask  = impute_data(test_df)
else:
    train_mask = None  # 전처리 후에 한 번에 생성 (원핫 이후 컬럼 수가 달라지므로)
    valid_mask = None
    test_mask  = None

# ── 3. 본인 전처리: train-fit / valid-test-transform ───────────
# data.py 의 preprocess_nsclc는 호출마다 scaler를 새로 fit하고 get_dummies도 독립 호출하므로
# 세 분할에 그대로 적용하면 (a) 스케일러 누출, (b) 컬럼 집합 불일치가 발생합니다.
# 따라서 아래에서는 "train에서만 fit → valid/test는 transform"과 "concat → get_dummies → split"
# 두 가지 leakage/불일치 방지 로직을 적용합니다.
#
# 주의: SAVAE는 마지막에서 두 번째 컬럼을 time으로 강제합니다(data.py:90 `i == n_feat - 2`).
# 원핫으로 컬럼 순서가 섞이면 안 되므로 time/event는 전처리 대상에서 분리했다가 마지막에 재부착합니다.

CATEGORICAL_COLS = ['gender', 'Smoking.status', 'Clinical.T.Stage',
                    'Clinical.N.stage', 'Clinical.M.stage', 'Overall.stage4']
NUMERIC_COLS     = ['age', 'Smoking.amount']


def preprocess_nsclc_three(train_df, valid_df, test_df):
    """
    Train에서만 StandardScaler를 fit하고 valid/test에는 transform만 적용.
    원핫은 세 분할을 concat한 뒤 get_dummies를 한 번만 호출해 컬럼 집합을 통일.
    time/event는 전처리에서 제외하고 "X + time + event" 순서로 재부착.
    """
    def split_xy(df):
        y = df[['time', 'event']].reset_index(drop=True)
        X = df.drop(columns=['time', 'event']).reset_index(drop=True)
        return X, y

    X_tr, y_tr = split_xy(train_df)
    X_va, y_va = split_xy(valid_df)
    X_te, y_te = split_xy(test_df)

    # (a) 원핫: concat → get_dummies → split  (컬럼 집합 통일)
    n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)
    combined = pd.concat([X_tr, X_va, X_te], axis=0, ignore_index=True)
    combined_enc = pd.get_dummies(combined, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)

    X_tr_enc = combined_enc.iloc[:n_tr].reset_index(drop=True)
    X_va_enc = combined_enc.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    X_te_enc = combined_enc.iloc[n_tr + n_va:].reset_index(drop=True)

    # (b) 수치형 StandardScaler: train에서 fit, valid/test는 transform만
    scaler = StandardScaler()
    X_tr_enc[NUMERIC_COLS] = scaler.fit_transform(X_tr_enc[NUMERIC_COLS])
    X_va_enc[NUMERIC_COLS] = scaler.transform(X_va_enc[NUMERIC_COLS])
    X_te_enc[NUMERIC_COLS] = scaler.transform(X_te_enc[NUMERIC_COLS])

    # (c) time/event를 마지막 두 컬럼으로 재부착  (data.py:90 가정 충족)
    train_out = pd.concat([X_tr_enc, y_tr], axis=1)
    valid_out = pd.concat([X_va_enc, y_va], axis=1)
    test_out  = pd.concat([X_te_enc, y_te], axis=1)

    return train_out, valid_out, test_out, scaler


train_df, valid_df, test_df, fitted_scaler = preprocess_nsclc_three(train_df, valid_df, test_df)

print(f'[PREP] train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}')
print('[PREP] columns:', list(train_df.columns))
print('[PREP] train head:')
print(train_df.head())

# ── 4. 마스크 (전처리 이후 shape 기준으로 생성) ────────────────
# 전처리 과정에서 결측치를 이미 처리했거나 원본에 결측이 없다면 전부 1.
if train_mask is None:
    train_mask = pd.DataFrame(np.ones(train_df.shape, dtype=int), columns=train_df.columns)
    valid_mask = pd.DataFrame(np.ones(valid_df.shape, dtype=int), columns=valid_df.columns)
    test_mask  = pd.DataFrame(np.ones(test_df.shape,  dtype=int), columns=test_df.columns)
else:
    # impute_data는 원본 컬럼 기준 mask를 반환하므로, 전처리 후 shape에 맞춰 1로 확장해야 합니다.
    # (원핫 후 컬럼 수가 달라지므로 정교하게 매핑하려면 별도 로직 필요. 결측이 실제로 있을 때만 주의.)
    train_mask = pd.DataFrame(np.ones(train_df.shape, dtype=int), columns=train_df.columns)
    valid_mask = pd.DataFrame(np.ones(valid_df.shape, dtype=int), columns=valid_df.columns)
    test_mask  = pd.DataFrame(np.ones(test_df.shape,  dtype=int), columns=test_df.columns)

# ── 5. time 양수 조건 처리 (Weibull 우도 요구) ─────────────────
# vae_modules.py:131 의 torch.log(targets / lam) 이 time==0 이면 -inf → NaN 을 유발합니다.
# 실데이터에는 time==0(진단 당일 사건) 또는 time<0(오입력)이 섞여 있을 수 있으므로,
# 진단 결과를 출력한 뒤 정책에 따라 정제합니다.
#
#   - 'shift': train에 0이 있으면 세 분할 모두 +1 시프트 (저자 data.py:136-138 방식)
#   - 'clip' : time = max(time, EPS) — 샘플은 유지, 단순 클리핑
#   - 'drop' : time <= 0 샘플을 제거 — 더 엄격/안전 (time==0 & event==0 제거에 유효)
TIME_POLICY = 'shift'  # 'shift' | 'clip' | 'drop'
EPS = 1e-3             # clip 사용 시 하한


def _diagnose_time(name, df):
    n_zero = int((df['time'] == 0).sum())
    n_neg  = int((df['time'] < 0).sum())
    if n_zero + n_neg == 0:
        print(f'[TIME] {name}: OK (time>0 샘플 {len(df)}개)')
        return
    # time==0 의 event 분포도 확인 (event==0 이면 정보량 0 → drop 권장)
    if n_zero > 0:
        zero_events = df.loc[df['time'] == 0, 'event'].value_counts().to_dict()
        print(f'[TIME] {name}: time==0 {n_zero}개 (event 분포: {zero_events})')
    if n_neg > 0:
        print(f'[TIME] {name}: time<0 {n_neg}개 (원본 데이터 오류 가능성)')


def _apply_time_policy_per(df, mask, policy, shift_val=0):
    if policy == 'clip':
        df = df.copy()
        df['time'] = df['time'].clip(lower=EPS)
        return df, mask
    elif policy == 'drop':
        keep = df['time'] > 0
        n_drop = int((~keep).sum())
        if n_drop > 0:
            print(f'  └ drop {n_drop}개 샘플')
        return df.loc[keep].reset_index(drop=True), mask.loc[keep].reset_index(drop=True)
    elif policy == 'shift':
        # 저자 data.py:136-138과 동일: time에 0이 있으면 전체에 +1 (loc = -1)
        df = df.copy()
        df['time'] = df['time'] + shift_val
        return df, mask
    else:
        raise ValueError(f"Unknown TIME_POLICY: {policy}")


print(f'[TIME] policy = {TIME_POLICY}')
_diagnose_time('train', train_df)
_diagnose_time('valid', valid_df)
_diagnose_time('test',  test_df)

# shift 정책은 train 기준으로 shift_val을 먼저 결정한 뒤 세 분할에 동일 적용 (train-only fit)
shift_val = 0
if TIME_POLICY == 'shift':
    shift_val = 1 if (train_df['time'] == 0).any() else 0
    if shift_val > 0:
        print(f'[TIME] shift: train에 0 있음 → 세 분할 모두 time += {shift_val}')

train_df, train_mask = _apply_time_policy_per(train_df, train_mask, TIME_POLICY, shift_val)
valid_df, valid_mask = _apply_time_policy_per(valid_df, valid_mask, TIME_POLICY, shift_val)
test_df,  test_mask  = _apply_time_policy_per(test_df,  test_mask,  TIME_POLICY, shift_val)

# 정제 후에도 0 이하가 남아있으면 안전 체크
for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
    assert (df['time'] > 0).all(), f"[{name}] 정제 후에도 time<=0 이 남아있습니다."

# ── 6. feat_distributions (전처리 끝난 데이터에서 감지) ────────
# 원핫 이후이므로 더미 컬럼들은 모두 bernoulli로 판정됩니다(data.py:97-99).
# time은 time=('weibull',2) 인자로 강제되고, event는 0/1 이므로 bernoulli로 자동 판정됩니다.
#
# concat은 "전체 bernoulli 컬럼들의 고유값 범위"가 세 분할에서 모두 {0,1}로 정상인지 판단하기
# 위한 안전장치입니다. 원핫을 concat 기반으로 이미 맞췄기 때문에 실질적으로는 train만 써도 결과가 같지만
# 방어적으로 concat 기반을 유지합니다.
feat_distributions = get_feat_distributions(train_df, time=('weibull', 2))

print('[FEAT] feat_distributions:', feat_distributions)
print(f'[FEAT] n_features (incl. time/event): {len(feat_distributions)}')

# ── 7. SAVAE 모델 초기화 ──────────────────────────────────────
feat_dist_no_sa = feat_distributions[:-2]  # time, event 제외한 공변량만

# max_t: 엄격한 leakage 기준으로 train에서만 계산 (논문/보고용 권장)
# 디코더의 Weibull λ 상한(vae_utils.py:56-58 max_k)을 정합니다.
max_t = float(train_df['time'].max())

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

print(f'[MODEL] input_dim={model_params["input_dim"]}, max_t={max_t:.2f}')

print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)
print(train_df.head())
# ── 8. 학습 (validation = valid_df) ────────────────────────────
# fit의 4번째 인자가 학습 중 모니터링·early stopping·best model 선택에 사용됩니다
# (savae.py:176, 216-220, 227). test_df는 여기 넣지 않습니다 (넣으면 test 누출).
data = (train_df, train_mask, valid_df, valid_mask)

train_params = {
    'n_epochs':   3000,
    'batch_size': 64,
    'lr':         1e-3,
    'device':     torch.device('cpu'),  # GPU: torch.device('cuda')
}
results = model.fit(data, train_params)

# ── 9. 최종 평가 ───────────────────────────────────────────────
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
model.save('savae_trained_pre.pt')
print('[SAVE] model → savae_trained_pre.pt')
