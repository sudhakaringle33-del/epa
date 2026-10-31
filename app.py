"""
Employee Performance Prediction - Gradio Web App (real IBM HR Analytics dataset)
Setup : put  WA_Fn-UseC_-HR-Employee-Attrition.csv  in the same folder as this file
Run   : python app.py     then open  http://127.0.0.1:7860
Models: Logistic Regression, Random Forest, Gradient Boosting (trained on startup)

TARGET: the IBM dataset's PerformanceRating has only 2 values (3 / 4), so a 3-class
"Performance Index" is built from real columns and split into Low / Medium / High:
   index = z(PerformanceRating) + z(PercentSalaryHike) + z(JobInvolvement)
         + z(JobSatisfaction) + z(TrainingTimesLastYear) - z(YearsSinceLastPromotion)
PerformanceRating and PercentSalaryHike are HIDDEN from the models (they are appraisal
outcomes; using them as inputs would just leak the answer).
"""
import os, sys
import numpy as np, pandas as pd, gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

CSV = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CLASSES = ["Low", "Medium", "High"]
if not os.path.exists(CSV):
    sys.exit(f"Dataset not found. Put '{CSV}' in the same folder as app.py and run again.")

# ---------- 1. Load real data + build the 3-class Performance Index ----------
raw = pd.read_csv(CSV)
z = lambda s: (s - s.mean()) / s.std()
index = (z(raw.PerformanceRating) + z(raw.PercentSalaryHike) + z(raw.JobInvolvement)
         + z(raw.JobSatisfaction) + z(raw.TrainingTimesLastYear) - z(raw.YearsSinceLastPromotion))
y = pd.qcut(index.rank(method="first"), 3, labels=CLASSES).astype(str)

# Input columns (protected attributes such as Gender / MaritalStatus are deliberately left out)
NUM = ["Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
       "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "RelationshipSatisfaction",
       "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
       "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
CAT = ["Department", "OverTime", "BusinessTravel"]
OPTS = {c: sorted(raw[c].unique()) for c in CAT}


# ---------- 2. Feature engineering ----------
def features(d, engineered=True):
    d = d[NUM + CAT].copy()
    if engineered:
        d["promotion_gap"] = d.YearsSinceLastPromotion / (d.YearsAtCompany + 1)
        d["role_stability"] = d.YearsInCurrentRole / (d.YearsAtCompany + 1)
        d["income_per_exp_year"] = d.MonthlyIncome / (d.TotalWorkingYears + 1)
        d["job_hopping"] = d.NumCompaniesWorked / (d.TotalWorkingYears + 1)
        d["training_per_year"] = d.TrainingTimesLastYear / (d.YearsAtCompany + 1)
        d["satisfaction_index"] = d[["JobSatisfaction", "EnvironmentSatisfaction",
                                     "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    for c in CAT:
        d[c] = pd.Categorical(d[c], categories=OPTS[c])
    return pd.get_dummies(d, columns=CAT, drop_first=True).astype(float)


# ---------- 3. Train models (with and without feature engineering) ----------
def make_models():
    return {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                            max_depth=2, subsample=0.8, random_state=42)}

idx_tr, idx_te = train_test_split(raw.index, test_size=0.2, stratify=y, random_state=42)
MODELS, rows = {}, []
for name in make_models():
    acc = {}
    for fe in (False, True):
        X = features(raw, fe)
        m = make_models()[name].fit(X.loc[idx_tr], y[idx_tr])
        acc[fe] = round(m.score(X.loc[idx_te], y[idx_te]), 3)
        if fe:
            MODELS[name] = m
    rows.append({"Model": name, "Original features": acc[False], "With feature engineering": acc[True]})
ACC = pd.DataFrame(rows)
XCOLS = features(raw).columns


# ---------- 4. App functions ----------
def predict(model_name, *vals):
    row = pd.DataFrame([dict(zip(NUM + CAT, vals))])
    model = MODELS[model_name]
    probs = model.predict_proba(features(row)[XCOLS])[0]
    r = row.iloc[0]; tips = []
    if r.OverTime == "Yes": tips.append("Works overtime - a burnout / attrition risk factor.")
    if r.YearsSinceLastPromotion >= 5: tips.append("Long time since last promotion - possible stagnation.")
    if r.TrainingTimesLastYear <= 1: tips.append("Little training last year - more training is linked to a higher index.")
    if r.JobSatisfaction <= 2: tips.append("Low job satisfaction.")
    if r.JobInvolvement <= 2: tips.append("Low job involvement.")
    return dict(zip(model.classes_, map(float, probs))), "\n".join(tips) or "No obvious risk factors flagged."


def importance_plot():
    rf = MODELS["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=XCOLS).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="#2980b9"); ax.set_title("Top features (Random Forest)")
    fig.tight_layout()
    return fig


# ---------- 5. UI ----------
med = raw[NUM].median()
def sl(col, lo, hi, label=None, step=1):
    return gr.Slider(lo, hi, value=float(med[col]), step=step, label=label or col)

with gr.Blocks(title="Employee Performance Prediction") as demo:
    gr.Markdown("# Employee Performance Prediction\nIBM HR Analytics dataset. Enter employee details to predict "
                "the performance class (**Low / Medium / High**).")
    with gr.Tab("Predict"):
        model = gr.Radio(list(MODELS), value="Random Forest", label="Model")
        with gr.Row():
            with gr.Column():
                age = sl("Age", 18, 60); dist = sl("DistanceFromHome", 1, 29, "Distance from home (km)")
                edu = sl("Education", 1, 5, "Education (1-5)"); level = sl("JobLevel", 1, 5, "Job level (1-5)")
                income = gr.Number(float(med.MonthlyIncome), label="Monthly income")
                ncomp = sl("NumCompaniesWorked", 0, 9, "Companies worked at before")
                stock = sl("StockOptionLevel", 0, 3, "Stock option level")
                twy = sl("TotalWorkingYears", 0, 40, "Total working years")
                yac = sl("YearsAtCompany", 0, 40, "Years at company")
            with gr.Column():
                yicr = sl("YearsInCurrentRole", 0, 18, "Years in current role")
                yslp = sl("YearsSinceLastPromotion", 0, 15, "Years since last promotion")
                ywcm = sl("YearsWithCurrManager", 0, 17, "Years with current manager")
                train = sl("TrainingTimesLastYear", 0, 6, "Training sessions last year")
                invol = sl("JobInvolvement", 1, 4, "Job involvement (1-4)")
                jsat = sl("JobSatisfaction", 1, 4, "Job satisfaction (1-4)")
                esat = sl("EnvironmentSatisfaction", 1, 4, "Environment satisfaction (1-4)")
                rsat = sl("RelationshipSatisfaction", 1, 4, "Relationship satisfaction (1-4)")
                wlb = sl("WorkLifeBalance", 1, 4, "Work-life balance (1-4)")
        with gr.Row():
            dept = gr.Dropdown(OPTS["Department"], value="Research & Development", label="Department")
            ot = gr.Radio(OPTS["OverTime"], value="No", label="Overtime")
            travel = gr.Dropdown(OPTS["BusinessTravel"], value="Travel_Rarely", label="Business travel")
        btn = gr.Button("Predict Performance", variant="primary")
        out_label = gr.Label(label="Predicted performance")
        out_tips = gr.Textbox(label="Insights", lines=3)
        # order must match NUM + CAT
        inputs = {"Age": age, "DistanceFromHome": dist, "Education": edu, "EnvironmentSatisfaction": esat,
                  "JobInvolvement": invol, "JobLevel": level, "JobSatisfaction": jsat, "MonthlyIncome": income,
                  "NumCompaniesWorked": ncomp, "RelationshipSatisfaction": rsat, "StockOptionLevel": stock,
                  "TotalWorkingYears": twy, "TrainingTimesLastYear": train, "WorkLifeBalance": wlb,
                  "YearsAtCompany": yac, "YearsInCurrentRole": yicr, "YearsSinceLastPromotion": yslp,
                  "YearsWithCurrManager": ywcm, "Department": dept, "OverTime": ot, "BusinessTravel": travel}
        btn.click(predict, [model] + [inputs[c] for c in NUM + CAT], [out_label, out_tips])
    with gr.Tab("Model comparison"):
        gr.Markdown("Test accuracy on a held-out 20% split (random guessing = 0.33).")
        gr.Dataframe(ACC, interactive=False)
        gr.Plot(importance_plot)
    with gr.Tab("About the target"):
        gr.Markdown("The dataset's `PerformanceRating` has only two values (3 and 4), so a **Performance Index** "
                    "was built from real columns (rating, salary hike, job involvement, job satisfaction, training, "
                    "years since promotion) and split into Low / Medium / High terciles. The rating and salary "
                    "hike are hidden from the models so they cannot simply read the answer back.")

if __name__ == "__main__":
    # 0.0.0.0 + PORT make it reachable on cloud servers/containers; locally it still opens on port 7860
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
