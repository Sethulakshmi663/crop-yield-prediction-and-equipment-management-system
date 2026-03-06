"""
Trains:
  1. Crop Recommender  (Random Forest Classifier)
  2. Yield Predictor   (Random Forest Regressor)

"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "crop_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
GRAPH_DIR  = os.path.join(BASE_DIR, "static", "graphs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ─── Palette ──────────────────────────────────────────────────────────────────
PALETTE  = ["#2ecc71","#27ae60","#1abc9c","#16a085","#f39c12","#e67e22","#e74c3c"]
ACCENT   = "#2ecc71"
BG       = "#0d1117"
FG       = "#ffffff"
GRID_CLR = "#21262d"

def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID_CLR, "axes.labelcolor": FG,
        "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": GRID_CLR,
        "axes.grid": True, "grid.alpha": 0.3,
        "font.family": "DejaVu Sans"
    })

apply_dark_theme()

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════
print("📂  Loading dataset …")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Season"] = df["Season"].str.strip()
df["Crop"]   = df["Crop"].str.strip()
df["State"]  = df["State"].str.strip()

# Drop rows with missing values
df.dropna(inplace=True)

# Remove extreme outliers in Yield using IQR per crop
def remove_outliers_df(df):
    result = []
    for crop, group in df.groupby("Crop"):
        Q1 = group["Yield"].quantile(0.05)
        Q3 = group["Yield"].quantile(0.95)
        IQR = Q3 - Q1
        mask = (group["Yield"] >= Q1 - 1.5*IQR) & (group["Yield"] <= Q3 + 1.5*IQR)
        result.append(group[mask])
    return pd.concat(result).reset_index(drop=True)

df = remove_outliers_df(df)

print(f"   Shape after cleaning: {df.shape}")
print(f"   Unique crops  : {df['Crop'].nunique()}")
print(f"   Unique seasons: {df['Season'].nunique()}")
print(f"   Unique states : {df['State'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  ENCODE LABELS
# ══════════════════════════════════════════════════════════════════════════════
le_crop   = LabelEncoder()
le_season = LabelEncoder()
le_state  = LabelEncoder()

df["Crop_enc"]   = le_crop.fit_transform(df["Crop"])
df["Season_enc"] = le_season.fit_transform(df["Season"])
df["State_enc"]  = le_state.fit_transform(df["State"])

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURES
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = ["Season_enc","State_enc","Area","Annual_Rainfall","Fertilizer","Pesticide"]

X = df[FEATURES].values
y_crop  = df["Crop_enc"].values
y_yield = df["Yield"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, yc_tr, yc_te = train_test_split(X_scaled, y_crop,  test_size=0.2, random_state=42, stratify=y_crop)
X_tr2,X_te2,yy_tr,yy_te  = train_test_split(X_scaled, y_yield, test_size=0.2, random_state=42)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN CROP RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
print("\n🌱  Training Crop Recommender (Random Forest Classifier) …")
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42,
                             n_jobs=-1, class_weight="balanced")
clf.fit(X_tr, yc_tr)
yc_pred = clf.predict(X_te)
clf_acc  = (yc_pred == yc_te).mean()
print(f"   Accuracy : {clf_acc*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN YIELD PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊  Training Yield Predictor (Random Forest Regressor) …")
X_yield_feats = ["Season_enc","State_enc","Crop_enc","Area","Annual_Rainfall","Fertilizer","Pesticide"]
Xy = df[X_yield_feats].values
scalery = StandardScaler()
Xy_scaled = scalery.fit_transform(Xy)
Xy_tr,Xy_te,yy_tr,yy_te = train_test_split(Xy_scaled, y_yield, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
reg.fit(Xy_tr, yy_tr)
yy_pred = reg.predict(Xy_te)
r2  = r2_score(yy_te, yy_pred)
mae = mean_absolute_error(yy_te, yy_pred)
rmse= np.sqrt(mean_squared_error(yy_te, yy_pred))
print(f"   R²   : {r2:.4f}")
print(f"   MAE  : {mae:.4f}")
print(f"   RMSE : {rmse:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n💾  Saving models …")
pickle.dump(clf,      open(os.path.join(MODEL_DIR,"crop_recommender.pkl"),"wb"))
pickle.dump(reg,      open(os.path.join(MODEL_DIR,"yield_predictor.pkl"),"wb"))
pickle.dump(le_crop,  open(os.path.join(MODEL_DIR,"le_crop.pkl"),"wb"))
pickle.dump(le_season,open(os.path.join(MODEL_DIR,"le_season.pkl"),"wb"))
pickle.dump(le_state, open(os.path.join(MODEL_DIR,"le_state.pkl"),"wb"))
pickle.dump(scaler,   open(os.path.join(MODEL_DIR,"scaler.pkl"),"wb"))
pickle.dump(scalery,  open(os.path.join(MODEL_DIR,"scaler_yield.pkl"),"wb"))

# Save metadata
meta = {
    "crops":   list(le_crop.classes_),
    "seasons": list(le_season.classes_),
    "states":  list(le_state.classes_),
    "features": FEATURES,
    "yield_features": X_yield_feats,
    "classifier_accuracy": round(clf_acc*100, 2),
    "regressor_r2":  round(r2, 4),
    "regressor_mae": round(mae, 4),
    "regressor_rmse":round(rmse, 4),
}
pickle.dump(meta, open(os.path.join(MODEL_DIR,"meta.pkl"),"wb"))
print("   Models saved ✔")


# ── Graph 7: Actual vs Predicted Yield ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,8))
sample_idx = np.random.choice(len(yy_te), min(1500, len(yy_te)), replace=False)
ax.scatter(yy_te[sample_idx], yy_pred[sample_idx],
           alpha=0.3, s=12, color=ACCENT, label="Predictions")
lims = [min(yy_te.min(), yy_pred.min()), max(yy_te.max(), yy_pred.max())]
ax.plot(lims, lims, color="#e74c3c", linewidth=1.5, label="Perfect Fit")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"Actual vs Predicted Yield  (R²={r2:.3f})", fontsize=14, color=ACCENT, fontweight="bold")
ax.set_xlabel("Actual Yield", color=FG)
ax.set_ylabel("Predicted Yield", color=FG)
ax.legend(facecolor=BG, edgecolor=GRID_CLR)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,"actual_vs_predicted.png"), dpi=120)
plt.close()
print("   ✔ actual_vs_predicted.png")

# ── Graph 8: Yearly Trend of Area Cultivated ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12,5))
yearly = df.groupby("Crop_Year")["Area"].sum()
ax.fill_between(yearly.index, yearly.values, alpha=0.3, color=ACCENT)
ax.plot(yearly.index, yearly.values, color=ACCENT, linewidth=2)
ax.set_title("Yearly Trend of Total Area Cultivated", fontsize=14, color=ACCENT, fontweight="bold")
ax.set_xlabel("Year", color=FG)
ax.set_ylabel("Total Area (hectares)", color=FG)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,"yearly_area_trend.png"), dpi=120)
plt.close()
print("   ✔ yearly_area_trend.png")

# ── Graph 9: Crop Distribution Pie ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,8))
top_n = df["Crop"].value_counts().nlargest(10)
other = df["Crop"].value_counts().iloc[10:].sum()
labels = list(top_n.index) + ["Others"]
sizes  = list(top_n.values) + [other]
explode = [0.05]*len(labels)
wedge_props = {"linewidth": 0.5, "edgecolor": BG}
ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140,
       colors=plt.cm.Set2.colors[:len(labels)],
       explode=explode, wedgeprops=wedge_props,
       textprops={"color": FG, "fontsize": 9})
ax.set_title("Crop Distribution in Dataset", fontsize=14, color=ACCENT, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR,"crop_distribution.png"), dpi=120)
plt.close()
print("   ✔ crop_distribution.png")

print("\n✅  All done! Models and graphs are ready.")
print(f"\n   Classifier Accuracy : {clf_acc*100:.2f}%")
print(f"   Regressor R²        : {r2:.4f}")
print(f"   Regressor MAE       : {mae:.4f}")
