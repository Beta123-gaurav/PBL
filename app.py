
---

## **3. app.py**
*NOTE: Replace any references to `Synthetic_health_lifestyle_dataset.csv` with your actual file if it differs. Adjust column names as needed!*

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

# Set page config
st.set_page_config(page_title="Health Analytics Dashboard", layout="wide")

# Sidebar: File uploader and downloaders
st.sidebar.title("📦 Upload or Download Data")
data_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
if data_file:
    df = pd.read_csv(data_file)
else:
    df = pd.read_csv("Synthetic_health_lifestyle_dataset.csv")

if st.sidebar.button("Download current data as CSV"):
    st.sidebar.download_button(
        label="Download Data",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="current_data.csv",
        mime="text/csv",
    )

# Encode categorical columns for modeling
df_display = df.copy()
le_dict = {}
for col in df_display.select_dtypes(include='object'):
    le = LabelEncoder()
    df_display[col] = le.fit_transform(df_display[col].astype(str))
    le_dict[col] = le

# --- Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Visualization",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
])

# ========================== TAB 1: Data Visualization ==========================
with tab1:
    st.header("Data Visualization & Descriptive Insights")
    st.markdown("Explore 10+ complex visual insights from your dataset with filters below.")
    
    # --- Filters
    with st.expander("Filters", expanded=True):
        min_age, max_age = st.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (20,60))
        gender_sel = st.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
        smoke_sel = st.multiselect("Smoking", options=df['Smoking'].unique(), default=list(df['Smoking'].unique()))
    filt_df = df[
        (df['Age']>=min_age) & (df['Age']<=max_age) &
        (df['Gender'].isin(gender_sel)) &
        (df['Smoking'].isin(smoke_sel))
    ]
    st.dataframe(filt_df.head(15))

    # 1. Chronic Disease Pie
    st.markdown("**Chronic Disease Distribution**")
    fig1, ax1 = plt.subplots()
    filt_df['Chronic_Disease'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
    st.pyplot(fig1)
    st.caption("Shows percentage of people with and without chronic disease.")

    # 2. Age Histogram
    st.markdown("**Age Distribution**")
    st.plotly_chart(px.histogram(filt_df, x="Age", color="Chronic_Disease", nbins=25))
    st.caption("Age distribution colored by disease presence.")

    # 3. Gender Bar
    st.markdown("**Gender Proportion**")
    st.bar_chart(filt_df['Gender'].value_counts())
    st.caption("Breakdown of gender in filtered data.")

    # 4. BMI Boxplot by Disease
    st.markdown("**BMI by Chronic Disease**")
    st.plotly_chart(px.box(filt_df, x="Chronic_Disease", y="BMI", color="Chronic_Disease"))
    st.caption("BMI ranges per disease status.")

    # 5. Alcohol vs Disease
    st.markdown("**Alcohol Consumption vs Chronic Disease**")
    st.plotly_chart(px.box(filt_df, x="Chronic_Disease", y="Alcohol Consumption"))
    st.caption("Alcohol habits across disease status.")

    # 6. Smoking vs Disease (Bar)
    st.markdown("**Smoking Category by Disease**")
    st.plotly_chart(px.histogram(filt_df, x="Smoking", color="Chronic_Disease", barmode="group"))
    st.caption("Relation of smoking habit and disease.")

    # 7. Physical Activity Histogram
    st.markdown("**Physical Activity Distribution**")
    st.plotly_chart(px.histogram(filt_df, x="Physical Activity", nbins=30))
    st.caption("Physical activity scores in the data.")

    # 8. Sleep Hours by Disease
    st.markdown("**Sleep Hours by Chronic Disease**")
    st.plotly_chart(px.box(filt_df, x="Chronic_Disease", y="Sleep Hours", color="Chronic_Disease"))
    st.caption("Sleep patterns for each disease group.")

    # 9. Steps vs BMI Scatter
    st.markdown("**Steps per Day vs BMI**")
    st.plotly_chart(px.scatter(filt_df, x="Steps per Day", y="BMI", color="Chronic_Disease"))
    st.caption("Relation of daily steps with BMI.")

    # 10. Correlation Heatmap
    st.markdown("**Correlation Heatmap**")
    fig_corr, ax_corr = plt.subplots(figsize=(10,6))
    sns.heatmap(df_display.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
    st.caption("How numeric columns relate to each other.")

    # 11. Steps vs Age by Gender
    st.markdown("**Steps vs Age by Gender**")
    st.plotly_chart(px.scatter(filt_df, x="Age", y="Steps per Day", color="Gender"))
    st.caption("Activity variation by age/gender.")

    # 12. Alcohol by Gender Box
    st.markdown("**Alcohol Consumption by Gender**")
    st.plotly_chart(px.box(filt_df, x="Gender", y="Alcohol Consumption"))
    st.caption("Alcohol differences among genders.")

# ========================== TAB 2: Classification ==========================
with tab2:
    st.header("Classification (KNN, Decision Tree, Random Forest, GBRT)")
    target = "Chronic_Disease"
    features = [c for c in df.columns if c != target]
    st.markdown("Select features and run classification algorithms. See scores, confusion matrices, ROC curve, and make predictions on new data.")

    # --- Data encoding/prep
    X = df_display.drop(columns=[target])
    y = df_display[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Model configs
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
        "GBRT": GradientBoostingClassifier(random_state=0)
    }
    results = []
    y_score_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
        # For ROC
        try:
            y_score = model.predict_proba(X_test)[:,1]
        except:
            y_score = model.decision_function(X_test)
        y_score_dict[name] = y_score

        results.append({
            "Model": name,
            "Train Acc": accuracy_score(y_train, y_pred_train),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1": f1_score(y_test, y_pred, average='macro')
        })
    st.dataframe(pd.DataFrame(results).round(3))

    # Confusion Matrix Dropdown
    algo_sel = st.selectbox("Select model for confusion matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[algo_sel].predict(X_test))
    st.markdown(f"**Confusion Matrix: {algo_sel}**")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC Curve
    st.markdown("**ROC Curve (All Models)**")
    fig_roc, ax_roc = plt.subplots()
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, y_score_dict[name])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax_roc.plot([0,1],[0,1],'--',color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Upload new data to predict
    st.markdown("**Upload new data for prediction (same features, without target)**")
    pred_file = st.file_uploader("Upload new data (csv)", key="class_pred")
    if pred_file:
        new_X = pd.read_csv(pred_file)
        # Try encode categorical
        for col in new_X.columns:
            if col in le_dict:
                new_X[col] = le_dict[col].transform(new_X[col].astype(str))
        pred_algo = st.selectbox("Model to use for prediction", list(models.keys()), key="pred_model")
        preds = models[pred_algo].predict(new_X)
        pred_df = new_X.copy()
        pred_df["Predicted_Label"] = preds
        st.dataframe(pred_df)
        st.download_button("Download predictions as CSV", pred_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# ========================== TAB 3: Clustering ==========================
with tab3:
    st.header("Clustering (KMeans)")
    st.markdown("Change number of clusters and see cluster segmentation and customer personas.")

    clust_cols = st.multiselect("Columns to cluster on", list(df_display.columns), default=["Age","BMI","Alcohol Consumption","Physical Activity","Sleep Hours"])
    n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

    km = KMeans(n_clusters=n_clusters, random_state=0)
    X_clust = df_display[clust_cols]
    labels = km.fit_predict(X_clust)
    df_clust = df.copy()
    df_clust["Cluster"] = labels

    # Elbow chart
    sse = []
    for k in range(2,11):
        km_tmp = KMeans(n_clusters=k, random_state=0)
        sse.append(km_tmp.fit(X_clust).inertia_)
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(2,11), sse, marker="o")
    ax_elbow.set_xlabel("Number of Clusters")
    ax_elbow.set_ylabel("Inertia (SSE)")
    st.pyplot(fig_elbow)
    st.caption("Elbow plot helps select optimal cluster number.")

    # Cluster Persona Table
    persona = df_clust.groupby("Cluster")[clust_cols].mean().round(2)
    st.markdown("**Cluster Personas (Means by Cluster):**")
    st.dataframe(persona)

    # Download labelled data
    st.download_button(
        "Download cluster-labelled data",
        df_clust.to_csv(index=False).encode("utf-8"),
        "clustered_data.csv",
        "text/csv"
    )

    # Visualize clusters (for first 2 dims)
    st.markdown("**Cluster Scatter (first 2 dimensions):**")
    if len(clust_cols)>=2:
        fig_scat = px.scatter(df_clust, x=clust_cols[0], y=clust_cols[1], color="Cluster")
        st.plotly_chart(fig_scat)

# ========================== TAB 4: Association Rule Mining ==========================
with tab4:
    st.header("Association Rule Mining (Apriori)")
    st.markdown("Select columns and parameters to see top 10 association rules by confidence.")

    cat_cols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category'] and len(df[col].unique())<20]
    apri_cols = st.multiselect("Columns to use for apriori", cat_cols, default=cat_cols[:2])
    min_support = st.slider("Min Support", 0.01, 0.5, 0.1, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)

    # Prepare data for apriori (one-hot)
    if len(apri_cols)>=2:
        df_apri = pd.get_dummies(df[apri_cols])
        freq_items = apriori(df_apri, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
        st.caption("Top 10 rules filtered by confidence.")

# ========================== TAB 5: Regression ==========================
with tab5:
    st.header("Regression Models (Linear, Ridge, Lasso, DT)")
    st.markdown("Quick insights from applying regression algorithms to numeric targets.")

    reg_target = st.selectbox("Select Target Variable", ["BMI","Alcohol Consumption","Physical Activity","Sleep Hours"], index=0)
    reg_feats = st.multiselect("Features to use", [col for col in df_display.columns if col!=reg_target and col!=target], default=["Age","Steps per Day"])

    X_reg = df_display[reg_feats]
    y_reg = df_display[reg_target]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)
    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=0)
    }
    reg_results = []
    for name, reg in regs.items():
        reg.fit(Xr_train, yr_train)
        y_pred = reg.predict(Xr_test)
        reg_results.append({
            "Model": name,
            "Train R2": reg.score(Xr_train, yr_train),
            "Test R2": reg.score(Xr_test, yr_test)
        })
    st.dataframe(pd.DataFrame(reg_results).round(3))
    st.caption("Shows R2 scores for different regression models.")

    # Plot prediction vs actual for selected model
    reg_sel = st.selectbox("Model for plot", list(regs.keys()))
    y_pred_sel = regs[reg_sel].predict(Xr_test)
    fig_reg, ax_reg = plt.subplots()
    ax_reg.scatter(yr_test, y_pred_sel, alpha=0.6)
    ax_reg.set_xlabel("Actual")
    ax_reg.set_ylabel("Predicted")
    ax_reg.set_title(f"Actual vs Predicted: {reg_sel}")
    st.pyplot(fig_reg)
    st.caption("Prediction accuracy visualization.")

st.sidebar.info("Developed by [Your Name]. For HR & Data Science insights.")
