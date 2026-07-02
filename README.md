# Customer Segmentation — Machine Learning Project
## Mall Customers Dataset | K-Means Clustering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.1.4-green?style=flat-square&logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

*An end-to-end unsupervised Machine Learning project for grouping mall customers into actionable business segments.*

</div>

---

## 📌 Project Overview

This project applies **K-Means Clustering** to the Mall Customers dataset to identify distinct customer segments based on **Annual Income** and **Spending Score**. The goal is to help the marketing team understand customer behaviour and design personalised strategies to increase revenue, retention, and customer satisfaction.

---

## 🎯 Problem Statement

A mall wants to understand its diverse customer base more deeply. With 200 customer records, the team needs to:

1. **Identify natural customer groups** without predefined labels (unsupervised learning).
2. **Profile each group** by income and spending behaviour.
3. **Generate targeted marketing strategies** for each segment.

---

## 📂 Dataset Information

| Field | Details |
|---|---|
| **Name** | Mall Customers Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| **Rows** | 200 |
| **Columns** | 5 |
| **Target** | None (unsupervised) |

### Feature Descriptions

| Column | Type | Business Meaning |
|---|---|---|
| `CustomerID` | Integer | Unique identifier — not used in modelling |
| `Gender` | Categorical | Male / Female — useful for demographic targeting |
| `Age` | Integer | Customer age — guides age-specific campaigns |
| `Annual Income (k$)` | Integer | Estimated yearly income (thousands) — purchasing power proxy |
| `Spending Score (1–100)` | Integer | Mall-assigned score based on purchase frequency & amount |

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Pandas | 2.1.4 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Matplotlib | 3.8.2 | Static visualisations |
| Seaborn | 0.13.2 | Statistical plots |
| Scikit-learn | 1.4.0 | K-Means, StandardScaler, metrics |
| SciPy | 1.12.0 | Convex hull for cluster boundaries |
| Streamlit | 1.31.0 | Interactive web dashboard |
| Plotly | 5.18.0 | Interactive charts in the dashboard |

---

## 🔬 Methodology

```
Raw Data → EDA → Feature Selection → Feature Scaling → Optimal K Selection → K-Means → Analysis → Insights
```

### Step-by-Step

1. **Data Understanding** — Shape, dtypes, missing values, statistics.
2. **EDA** — Distributions, gender analysis, correlation heatmap, pairplot, scatter plots, boxplots.
3. **Preprocessing** — Selected `Annual Income` and `Spending Score`; applied `StandardScaler`.
4. **Optimal K** — Evaluated K=2…11 using Elbow Method, Silhouette Score, and Davies-Bouldin Index → **K=5 selected**.
5. **Model Training** — KMeans with `k-means++` initialisation, `n_init=10`, `random_state=42`.
6. **Visualisation** — Cluster scatter plots with centroids and convex-hull boundaries; before vs after comparison.
7. **Cluster Analysis** — Per-cluster statistics and business profiles.
8. **Recommendations** — Targeted marketing strategies per segment.
9. **Evaluation** — Silhouette Score, Inertia, Davies-Bouldin Index.

---

## 📊 Results

### The 5 Customer Segments

| Cluster | Name | Customers | Avg Income (k$) | Avg Spending |
|---|---|---|---|---|
| 0 | 💎 High Income – High Spending | ~39 | ~86 | ~82 |
| 1 | 🛒 Low Income – High Spending | ~35 | ~26 | ~78 |
| 2 | 🏪 Average Customers | ~81 | ~55 | ~50 |
| 3 | 🎯 High Income – Low Spending | ~23 | ~88 | ~17 |
| 4 | 💤 Low Income – Low Spending | ~22 | ~26 | ~21 |

### Model Quality Metrics

| Metric | Value | Interpretation |
|---|---|---|
| Silhouette Score | ~0.55 | Good cluster separation (> 0.5 is strong) |
| Davies-Bouldin Index | ~0.57 | Well-separated clusters (< 1.0 is good) |
| Inertia (WCSS) | ~445 | Compact within-cluster structure |

---

## 💡 Business Insights & Recommendations

### 💎 High Income – High Spending
> **The VIPs** — High revenue, high engagement.
- Launch Premium Loyalty Programme.
- Assign personal shoppers / relationship managers.
- Early access to new products and exclusive sale events.

### 🛒 Low Income – High Spending
> **The Enthusiastic Shoppers** — Love spending but budget-constrained.
- Promote value bundles and budget-friendly lines.
- Introduce BNPL (Buy Now Pay Later) plans.
- Gamified loyalty points to drive repeat visits.

### 🏪 Average Customers
> **The Core Base** — Stable revenue, moderate engagement.
- Cross-sell complementary products at checkout.
- Seasonal promotions aligned with paydays.
- Personalised email newsletters.

### 🎯 High Income – Low Spending
> **The Untapped Goldmine** — Can spend, but don't.
- Survey to uncover barriers and preferences.
- Experiential marketing events.
- Quality-focused, premium-positioning ads.

### 💤 Low Income – Low Spending
> **The Retention Challenge** — Infrequent, price-sensitive visitors.
- Deep-discount flash campaigns.
- Geo-targeted push notifications near the mall.
- Community engagement via social media.

---

## 🚀 How to Run

### Prerequisites

```bash
git clone <your-repo-url>
cd customer_segmentation
pip install -r requirements.txt
```

### Run the Python Script

```bash
python customer_segmentation.py
```

Outputs (charts + enriched CSV) will be saved to the `outputs/` directory.

### Launch the Streamlit Dashboard

```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501` with:
- Interactive EDA charts
- Optimal K visualisations
- Live cluster scatter plots (3D view included)
- Segment-level analysis table and radar chart
- Segment-specific marketing recommendations
- **Live Customer Predictor** — enter any income and spending score to instantly see which segment a customer belongs to.

---

## 📁 Project Structure

```
customer_segmentation/
├── data/
│   └── Mall_Customers.csv          # Dataset
├── outputs/                        # Auto-generated charts & enriched CSV
│   ├── eda_age_distribution.png
│   ├── eda_income_distribution.png
│   ├── eda_spending_distribution.png
│   ├── eda_gender_distribution.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_pairplot.png
│   ├── eda_income_vs_spending.png
│   ├── eda_boxplots.png
│   ├── cluster_selection_metrics.png
│   ├── clusters_main.png
│   ├── before_after_clustering.png
│   ├── cluster_analysis_bar.png
│   ├── model_evaluation.png
│   └── customers_with_clusters.csv
├── customer_segmentation.py        # Main pipeline script
├── streamlit_dashboard.py          # Interactive Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🔮 Future Scope

1. **Richer Feature Set** — Include Age and Gender with PCA/UMAP dimensionality reduction.
2. **Alternative Algorithms** — DBSCAN, Gaussian Mixture Models, Agglomerative Clustering.
3. **RFM Analysis** — Add Recency, Frequency, Monetary dimensions for deeper segmentation.
4. **Online Clustering** — Retrain monthly as new customer data arrives (Mini-Batch K-Means).
5. **Customer Lifetime Value (CLV)** — Combine segmentation with CLV prediction for prioritisation.
6. **API Deployment** — Wrap the model in a FastAPI endpoint for real-time segment prediction.
7. **A/B Testing Framework** — Measure campaign effectiveness per segment over time.

---

## 👩‍💻 Author

Built with ❤️ for a **GitHub portfolio project** and **ML interview preparation**.

> *"Data-driven segmentation is the foundation of modern personalised marketing."*

---

## 📜 License

MIT License — feel free to use, modify, and share this project.
