# crop-yield-prediction-and-equipment-management-system



## 🚀 What It Does

- 🌱 **Crop Recommendation** — Top 5 crops with probability scores based on season, state, rainfall, fertilizer & pesticide
- 📈 **Yield Prediction** — Predicted yield (t/ha) with 95% confidence interval and risk level (Low / Medium / High)
- ⚠️ **Smart Alerts** — Warns about low rainfall, flood risk, excess fertilizer, or high pesticide usage
- 💰 **Economic Analysis** — Estimates revenue, cost, profit and ROI from predicted yield
- 🚜 **Equipment Marketplace** — Browse, book, and review agricultural machinery for rent
- 👥 **Multi-Role System** — Separate dashboards for Farmers, Equipment Providers, and Admins

---

## 🤖 ML Models

- 🌿 **Crop Recommender** — Random Forest Classifier · 55 crops · 28 states · Accuracy: **48.65%**
- 📊 **Yield Predictor** — Random Forest Regressor · R² Score: **0.8961** · 95% CI output
- 🔢 **Dataset** — Kaggle Indian Crop Production · 6 seasons · 80/20 train-test split

---

## 🛠️ Tech Stack

- 🐍 **Backend** — Python · Flask
- 🧠 **ML** — scikit-learn · pandas · NumPy
- 🗄️ **Database** — SQLite
- 🎨 **Frontend** — HTML5 · CSS3 · JavaScript · Jinja2
- 🌦️ **API** — Open-Meteo Archive API (auto-fill rainfall)

---


## 👤 User Roles

| Role | Access |
|---|---|
| 🌾 Farmer | Predict · Recommend · Alerts · Economic Analysis · Book Equipment |
| 🛠️ Provider | List Equipment · Manage Bookings · Approve / Reject |
| 🔐 Admin | Manage Users · View All Data · Export CSV |


**Built with ❤️ for Indian farmers**

