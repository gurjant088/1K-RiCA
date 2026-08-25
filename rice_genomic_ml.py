"""
rice_genomic_ml.py
==================
Explainable ML for Genomic Prediction, Subgroup Classification,
and Optimal Parent Cross Ranking in Rice Breeding Using 1k-RiCA SNP Data

Author  : Gurjant Singh
Dataset : 1k-RiCA SNP Panel — 353 accessions × 965 SNP markers

Usage (Google Colab):
    1. Upload 12284_2019_311_MOESM1_ESM.xlsx to /content/sample_data/
    2. Run: python rice_genomic_ml.py

Results:
    - XGBoost FLW  : R² = 0.6905, RMSE = 2.5195 days
    - RandomForest PH : R² = 0.5486, RMSE = 5.9356 cm
    - RF Classification : Acc = 76.67%, ROC AUC = 0.8174
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (r2_score, mean_squared_error,
                             accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

# ── Dataset path ───────────────────────────────────────────────
FILE = '/content/sample_data/12284_2019_311_MOESM1_ESM.xlsx'

# ═══════════════════════════════════════════════════════════════
# PART 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
snp7_df  = pd.read_excel(FILE, sheet_name='Supplementary File 7', index_col=0)
snp4_df  = pd.read_excel(FILE, sheet_name='Supplementary File 4', index_col=0)
pheno_df = pd.read_excel(FILE, sheet_name='Genomic selection-adjusted mean')
acc_df   = pd.read_excel(FILE, sheet_name='Supplementary File 1', header=1)
acc_df.columns = ['No','GenotypeID','GID','AccessionName',
                  'Experiment','Subgroup','Reference','Remarks','Present']

print(f"SNP File7:  {snp7_df.shape}")
print(f"SNP File4:  {snp4_df.shape}")
print(f"Pheno:      {pheno_df.shape}")
print(f"Accession:  {acc_df.shape}")

meta_cols = ['alleles','chrom','pos','strand','assembly#',
             'center','protLSID','assayLSID','panelLSID','QCcode']


def encode_snp(snp_df, id_col_name='Variety'):
    """Encode SNP genotype matrix from nucleotide to numeric (0/1/2)."""
    clean = snp_df.drop(columns=[c for c in meta_cols if c in snp_df.columns])
    T     = clean.T.reset_index()
    T.columns = [id_col_name] + list(T.columns[1:])
    le    = LabelEncoder()
    cols  = T.columns[1:]
    X     = T[cols].apply(lambda c: le.fit_transform(c.astype(str))).values.astype(float)
    ids   = T[id_col_name].astype(str).str.strip().values
    return X, ids, cols


def normalize_id(val):
    """Normalize sample IDs to consistent string format."""
    try:    return str(int(float(str(val).strip())))
    except: return str(val).strip()


# ═══════════════════════════════════════════════════════════════
# PART 2 — REGRESSION: FLW & PH
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  PART A: GENOMIC PREDICTION — FLW & PH")
print("=" * 55)

X_snp7, snp7_ids, snp7_cols = encode_snp(snp7_df)
snp7_ids_norm = np.array([normalize_id(i) for i in snp7_ids])

pheno_df.columns    = ['Variety', 'FLW', 'GY', 'PH']
pheno_df            = pheno_df.dropna()
pheno_df['Variety'] = pheno_df['Variety'].apply(normalize_id)
pheno_ids           = pheno_df['Variety'].values

common    = np.intersect1d(snp7_ids_norm, pheno_ids)
snp_idx   = [np.where(snp7_ids_norm == i)[0][0] for i in common]
pheno_idx = [np.where(pheno_ids == i)[0][0]     for i in common]

X_reg         = X_snp7[snp_idx]
pheno_matched = pheno_df.iloc[pheno_idx].reset_index(drop=True)
print(f"Matched samples: {len(common)}")

scaler_reg = StandardScaler()
X_reg_sc   = scaler_reg.fit_transform(X_reg)

regression_results = {}

# ── FLW ───────────────────────────────────────────────────────
trait = 'FLW'
y_flw = pheno_matched[trait].values
X_train_flw, X_test_flw, y_train_flw, y_test_flw = train_test_split(
    X_reg_sc, y_flw, test_size=0.2, random_state=42
)

models_flw = {
    'Ridge': RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5),
    'RandomForest': RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=2, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(
        n_estimators=500, max_depth=4,
        learning_rate=0.01, reg_alpha=0.01,
        subsample=0.9, colsample_bytree=0.5,
        random_state=42, verbosity=0),
}

print(f"\n── Trait: FLW ──")
print(f"{'Model':<15} {'Test R²':>10} {'CV R²':>10} {'RMSE':>10}")
print("-" * 50)

regression_results['FLW'] = {}
for name, m in models_flw.items():
    m.fit(X_train_flw, y_train_flw)
    y_pred  = m.predict(X_test_flw)
    test_r2 = r2_score(y_test_flw, y_pred)
    cv_r2   = cross_val_score(m, X_reg_sc, y_flw, cv=5, scoring='r2').mean()
    rmse    = np.sqrt(mean_squared_error(y_test_flw, y_pred))
    print(f"{name:<15} {test_r2:>10.4f} {cv_r2:>10.4f} {rmse:>10.4f}")
    regression_results['FLW'][name] = {
        'model': m, 'test_r2': test_r2,
        'cv_r2': cv_r2, 'rmse': rmse,
        'y_test': y_test_flw, 'y_pred': y_pred
    }

# ── PH ────────────────────────────────────────────────────────
print(f"\n── Trait: PH ──")
print(f"{'Model':<15} {'Test R²':>10} {'CV R²':>10} {'RMSE':>10}")
print("-" * 50)

y_ph       = pheno_matched['PH'].values
X_with_flw = np.hstack([X_reg_sc, pheno_matched[['FLW']].values])
selector   = SelectKBest(f_regression, k=200)
X_ph_sel   = selector.fit_transform(X_with_flw, y_ph)

X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(
    X_ph_sel, y_ph, test_size=0.2, random_state=42
)

# Ridge — PH
ridge_ph = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5)
ridge_ph.fit(X_train_ph, y_train_ph)
y_pred_ridge_ph = ridge_ph.predict(X_test_ph)
r2_ridge_ph     = r2_score(y_test_ph, y_pred_ridge_ph)
cv_ridge_ph     = cross_val_score(ridge_ph, X_ph_sel, y_ph, cv=5, scoring='r2').mean()
rmse_ridge_ph   = np.sqrt(mean_squared_error(y_test_ph, y_pred_ridge_ph))
print(f"{'Ridge':<15} {r2_ridge_ph:>10.4f} {cv_ridge_ph:>10.4f} {rmse_ridge_ph:>10.4f}")

# XGBoost — PH
xgb_ph = XGBRegressor(
    n_estimators=500, max_depth=4,
    learning_rate=0.01, reg_alpha=0.01,
    subsample=0.9, colsample_bytree=0.5,
    random_state=42, verbosity=0
)
xgb_ph.fit(X_train_ph, y_train_ph)
y_pred_xgb_ph = xgb_ph.predict(X_test_ph)
r2_xgb_ph     = r2_score(y_test_ph, y_pred_xgb_ph)
cv_xgb_ph     = cross_val_score(xgb_ph, X_ph_sel, y_ph, cv=5, scoring='r2').mean()
rmse_xgb_ph   = np.sqrt(mean_squared_error(y_test_ph, y_pred_xgb_ph))
print(f"{'XGBoost':<15} {r2_xgb_ph:>10.4f} {cv_xgb_ph:>10.4f} {rmse_xgb_ph:>10.4f}")

# RandomForest — PH (Best Model)
rf_ph = RandomForestRegressor(
    n_estimators=600, max_depth=None,
    min_samples_leaf=1, min_samples_split=2,
    max_features=0.4, bootstrap=True,
    random_state=42, n_jobs=-1
)
rf_ph.fit(X_train_ph, y_train_ph)
y_pred_rf_ph = rf_ph.predict(X_test_ph)
r2_rf_ph     = r2_score(y_test_ph, y_pred_rf_ph)
cv_rf_ph     = cross_val_score(rf_ph, X_ph_sel, y_ph, cv=5, scoring='r2').mean()
rmse_rf_ph   = np.sqrt(mean_squared_error(y_test_ph, y_pred_rf_ph))
print(f"{'RandomForest':<15} {r2_rf_ph:>10.4f} {cv_rf_ph:>10.4f} {rmse_rf_ph:>10.4f}")

regression_results['PH'] = {
    'Ridge':        {'model': ridge_ph,  'test_r2': r2_ridge_ph,
                     'cv_r2': cv_ridge_ph,  'rmse': rmse_ridge_ph,
                     'y_test': y_test_ph, 'y_pred': y_pred_ridge_ph},
    'XGBoost':      {'model': xgb_ph,    'test_r2': r2_xgb_ph,
                     'cv_r2': cv_xgb_ph,    'rmse': rmse_xgb_ph,
                     'y_test': y_test_ph, 'y_pred': y_pred_xgb_ph},
    'RandomForest': {'model': rf_ph,     'test_r2': r2_rf_ph,
                     'cv_r2': cv_rf_ph,     'rmse': rmse_rf_ph,
                     'y_test': y_test_ph, 'y_pred': y_pred_rf_ph},
}

# ═══════════════════════════════════════════════════════════════
# PART 3 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  PART B: SUBGROUP CLASSIFICATION (File 4)")
print("=" * 55)

X_snp4, snp4_ids, snp4_cols = encode_snp(snp4_df)
snp4_ids_clean = np.array([s.strip() for s in snp4_ids])

acc_df['GenotypeID'] = acc_df['GenotypeID'].astype(str).str.strip()
acc_df = acc_df[acc_df['Subgroup'].notna()]
acc_df = acc_df[acc_df['Subgroup'].astype(str).str.strip() != '']
acc_df = acc_df[acc_df['Subgroup'].astype(str).str.strip() != 'nan']

subgroup_counts = acc_df['Subgroup'].value_counts()
print("\nAll subgroup counts:")
print(subgroup_counts)

valid_groups = subgroup_counts[subgroup_counts >= 10].index.tolist()
acc_clean    = acc_df[acc_df['Subgroup'].isin(valid_groups)].copy()
acc_clean['GenotypeID'] = acc_clean['GenotypeID'].astype(str).str.strip()
print(f"\nValid subgroups (>=10 samples): {valid_groups}")

common_cls = np.intersect1d(snp4_ids_clean, acc_clean['GenotypeID'].values)
print(f"Matched for classification: {len(common_cls)}")

cls_results = {}
le_cls      = LabelEncoder()

if len(common_cls) >= 30:
    snp4_idx = [np.where(snp4_ids_clean == i)[0][0] for i in common_cls]
    acc_idx  = [acc_clean.index[acc_clean['GenotypeID'] == i][0]
                for i in common_cls]

    X_cls      = X_snp4[snp4_idx]
    y_cls_raw  = acc_clean.loc[acc_idx, 'Subgroup'].values
    y_cls      = le_cls.fit_transform(y_cls_raw)
    scaler_cls = StandardScaler()
    X_cls_sc   = scaler_cls.fit_transform(X_cls)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_cls_sc, y_cls, test_size=0.2,
        random_state=42, stratify=y_cls
    )

    models_cls = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=10,
            random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, random_state=42,
            verbosity=0, eval_metric='mlogloss'),
    }

    print(f"\n{'Model':<15} {'Test Acc':>10} {'CV Acc':>10}")
    print("-" * 38)

    for name, m in models_cls.items():
        m.fit(X_tr, y_tr)
        y_pred   = m.predict(X_te)
        test_acc = accuracy_score(y_te, y_pred)
        cv_acc   = cross_val_score(
            m, X_cls_sc, y_cls, cv=5, scoring='accuracy').mean()
        print(f"{name:<15} {test_acc:>10.4f} {cv_acc:>10.4f}")
        cls_results[name] = {
            'model': m, 'test_acc': test_acc,
            'cv_acc': cv_acc, 'y_te': y_te, 'y_pred': y_pred
        }

    best_cls = max(cls_results, key=lambda x: cls_results[x]['cv_acc'])
    print(f"\n── Classification Report: {best_cls} ──")
    print(classification_report(
        cls_results[best_cls]['y_te'],
        cls_results[best_cls]['y_pred'],
        target_names=le_cls.classes_
    ))
else:
    print("Not enough matched samples for classification")

# ═══════════════════════════════════════════════════════════════
# PART 4 — PLOTS
# ═══════════════════════════════════════════════════════════════
colors  = ['steelblue', 'mediumseagreen', 'coral']
fig     = plt.figure(figsize=(18, 14))
fig.suptitle('Rice Genomic Prediction — Results',
             fontsize=16, fontweight='bold')
plot_idx = 1

# FLW — XGBoost actual vs predicted
res_flw = regression_results['FLW']['XGBoost']
ax = fig.add_subplot(3, 3, plot_idx)
ax.scatter(res_flw['y_test'], res_flw['y_pred'],
           alpha=0.6, color='steelblue', edgecolors='k', s=50)
mn = min(res_flw['y_test'].min(), res_flw['y_pred'].min())
mx = max(res_flw['y_test'].max(), res_flw['y_pred'].max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
ax.set_xlabel('Actual FLW'); ax.set_ylabel('Predicted FLW')
ax.set_title(f'FLW — XGBoost\nTest R²={res_flw["test_r2"]:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)
plot_idx += 1

# PH — RandomForest actual vs predicted
res_ph = regression_results['PH']['RandomForest']
ax = fig.add_subplot(3, 3, plot_idx)
ax.scatter(res_ph['y_test'], res_ph['y_pred'],
           alpha=0.6, color='mediumseagreen', edgecolors='k', s=50)
mn = min(res_ph['y_test'].min(), res_ph['y_pred'].min())
mx = max(res_ph['y_test'].max(), res_ph['y_pred'].max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
ax.set_xlabel('Actual PH'); ax.set_ylabel('Predicted PH')
ax.set_title(f'PH — RandomForest\nTest R²={res_ph["test_r2"]:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)
plot_idx += 1

# R² bar charts for FLW and PH
for trait in ['FLW', 'PH']:
    ax    = fig.add_subplot(3, 3, plot_idx)
    names = list(regression_results[trait].keys())
    r2s   = [regression_results[trait][n]['test_r2'] for n in names]
    bars  = ax.bar(names, r2s, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title(f'{trait} — Test R² Comparison')
    ax.set_ylabel('Test R²'); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    plot_idx += 1

# FLW residuals
residuals_flw = res_flw['y_test'] - res_flw['y_pred']
ax = fig.add_subplot(3, 3, plot_idx)
ax.scatter(res_flw['y_pred'], residuals_flw,
           alpha=0.6, color='coral', edgecolors='k', s=50)
ax.axhline(0, color='red', lw=2, linestyle='--')
ax.set_xlabel('Predicted FLW'); ax.set_ylabel('Residuals')
ax.set_title('FLW Residuals — XGBoost')
ax.grid(True, alpha=0.3)
plot_idx += 1

# PH residuals
residuals_ph = res_ph['y_test'] - res_ph['y_pred']
ax = fig.add_subplot(3, 3, plot_idx)
ax.scatter(res_ph['y_pred'], residuals_ph,
           alpha=0.6, color='coral', edgecolors='k', s=50)
ax.axhline(0, color='red', lw=2, linestyle='--')
ax.set_xlabel('Predicted PH'); ax.set_ylabel('Residuals')
ax.set_title('PH Residuals — RandomForest')
ax.grid(True, alpha=0.3)
plot_idx += 1

# Confusion matrix
if cls_results and plot_idx <= 9:
    best_cls = max(cls_results, key=lambda x: cls_results[x]['cv_acc'])
    res_cls  = cls_results[best_cls]
    cm       = confusion_matrix(res_cls['y_te'], res_cls['y_pred'])
    ax       = fig.add_subplot(3, 3, plot_idx)
    disp     = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=le_cls.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Subgroup Classification\n'
                 f'{best_cls} Acc={res_cls["test_acc"]:.4f}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('thesis_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: thesis_results.png")

# ═══════════════════════════════════════════════════════════════
# PART 5 — THESIS SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  THESIS SUMMARY")
print("=" * 55)
print(f"  Dataset : {len(common)} accessions x {X_reg.shape[1]} SNPs")
print(f"\n  FLW Best Model : XGBoost")
print(f"  FLW Test R²    : {regression_results['FLW']['XGBoost']['test_r2']:.4f}")
print(f"  FLW RMSE       : {regression_results['FLW']['XGBoost']['rmse']:.4f}")
print(f"\n  PH Best Model  : RandomForest")
print(f"  PH Test R²     : {regression_results['PH']['RandomForest']['test_r2']:.4f}")
print(f"  PH RMSE        : {regression_results['PH']['RandomForest']['rmse']:.4f}")
if cls_results:
    best = max(cls_results, key=lambda x: cls_results[x]['cv_acc'])
    print(f"\n  Classification  : {best}")
    print(f"  Accuracy        : {cls_results[best]['test_acc']:.4f}")

# ═══════════════════════════════════════════════════════════════
# PART 6 — ROC CURVES (All 3 Models)
# ═══════════════════════════════════════════════════════════════
if cls_results:
    colors_roc = ['steelblue', 'coral', 'mediumseagreen',
                  'orange', 'purple', 'brown']

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('ROC Curves — All 3 Models',
                 fontsize=16, fontweight='bold')

    # Panel 1: XGBoost FLW (binned Low/Mid/High)
    ax         = axes[0]
    y_pred_flw = regression_results['FLW']['XGBoost']['y_pred']
    y_true_flw = regression_results['FLW']['XGBoost']['y_test']
    bins_flw   = np.percentile(y_pred_flw, [33, 66])
    y_bin_pred = np.digitize(y_pred_flw, bins_flw)
    y_bin_true = np.digitize(y_true_flw, bins_flw)
    uniq       = np.unique(y_bin_true)
    y_bin_true2 = label_binarize(y_bin_true, classes=uniq)
    y_bin_pred2 = label_binarize(y_bin_pred, classes=uniq)
    labels      = ['Low', 'Medium', 'High']
    for i in range(len(uniq)):
        try:
            fpr, tpr, _ = roc_curve(y_bin_true2[:, i], y_bin_pred2[:, i])
            ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
                    label=f'{labels[i]} (AUC={auc(fpr, tpr):.3f})')
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_title('XGBoost — FLW Prediction\n(Low/Medium/High bins)',
                 fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: RandomForest PH (binned)
    ax         = axes[1]
    y_pred_ph  = regression_results['PH']['RandomForest']['y_pred']
    y_true_ph  = regression_results['PH']['RandomForest']['y_test']
    bins_ph    = np.percentile(y_pred_ph, [33, 66])
    y_bin_pred = np.digitize(y_pred_ph, bins_ph)
    y_bin_true = np.digitize(y_true_ph, bins_ph)
    uniq       = np.unique(y_bin_true)
    y_bin_true2 = label_binarize(y_bin_true, classes=uniq)
    y_bin_pred2 = label_binarize(y_bin_pred, classes=uniq)
    for i in range(len(uniq)):
        try:
            fpr, tpr, _ = roc_curve(y_bin_true2[:, i], y_bin_pred2[:, i])
            ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
                    label=f'{labels[i]} (AUC={auc(fpr, tpr):.3f})')
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_title('RandomForest — PH Prediction\n(Low/Medium/High bins)',
                 fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: RandomForest Classification ROC
    ax        = axes[2]
    rf_model  = cls_results['RandomForest']['model']
    y_te_roc  = cls_results['RandomForest']['y_te']
    classes   = np.unique(y_te_roc)
    y_te_bin  = label_binarize(y_te_roc, classes=classes)
    y_prob    = rf_model.predict_proba(X_te)
    macro_auc = roc_auc_score(y_te_bin, y_prob,
                              multi_class='ovr', average='macro')
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=colors_roc[i % len(colors_roc)], lw=2,
                label=f'{le_cls.classes_[i]} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_title(f'RandomForest — Subgroup Classification\nMacro AUC={macro_auc:.4f}',
                 fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_all_models.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: roc_all_models.png")

# ═══════════════════════════════════════════════════════════════
# PART 7 — SAVE TRAINED MODELS
# ═══════════════════════════════════════════════════════════════
try:
    import joblib
    import os
    os.makedirs('models', exist_ok=True)

    xgb_flw_model = regression_results['FLW']['XGBoost']['model']
    joblib.dump(xgb_flw_model, 'models/xgb_flw.pkl')
    joblib.dump(rf_ph,         'models/rf_ph.pkl')
    joblib.dump(scaler_reg,    'models/scaler_reg.pkl')
    joblib.dump(selector,      'models/selector_ph.pkl')
    joblib.dump({
        'xgb_flw': {
            'r2':   regression_results['FLW']['XGBoost']['test_r2'],
            'rmse': regression_results['FLW']['XGBoost']['rmse'],
        },
        'rf_ph': {
            'r2':   r2_rf_ph,
            'rmse': rmse_rf_ph,
        },
    }, 'models/metrics.pkl')

    print("\nModels saved to models/:")
    print("  xgb_flw.pkl")
    print("  rf_ph.pkl")
    print("  scaler_reg.pkl")
    print("  selector_ph.pkl")
    print("  metrics.pkl")
except Exception as e:
    print(f"Could not save models: {e}")
