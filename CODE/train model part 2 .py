
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
