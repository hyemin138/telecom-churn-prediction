"""EDA: 이탈 분포, 이탈 사유, 계약 형태, Offer, 연령대, 상관관계

실행: python src/01_eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import RESULTS, TARGET, load

OUT = RESULTS / "01_eda"
RED, BLUE = "#e74c3c", "#3498db"


def save(fig, name):
    """그래프 저장"""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def churn_rate(df, col):
    """범주별 이탈률(%)"""
    return (df[TARGET].eq("Yes").groupby(df[col]).mean() * 100).round(1)


def plot_distribution(df):
    """이탈 고객 비율 (26.5%)"""
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts, labels=counts.index, autopct="%.1f%%", colors=[BLUE, RED], startangle=90)
    ax.set_title("Churn Label Distribution")
    save(fig, "churn_distribution.png")


def plot_reasons(df):
    """이탈 사유 상위 10개"""
    top = df["Churn Reason"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=RED)
    ax.set(title="Top 10 Churn Reasons", xlabel="Count")
    save(fig, "churn_reasons.png")


def plot_rate(df, col, name, title, order=None):
    """범주별 이탈률 막대그래프"""
    rate = churn_rate(df, col)
    if order is not None:
        rate = rate.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(rate.index.astype(str), rate.values, color=RED)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set(title=title, ylabel="Churn rate (%)", ylim=(0, max(rate.values) * 1.2))
    save(fig, name)
    return rate


def plot_corr(df):
    """수치형 변수 상관계수 히트맵, |r| >= 0.5 변수쌍 표 저장"""
    num = df.select_dtypes("number").drop(columns=["Zip Code"])
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Heatmap (numeric)")
    save(fig, "corr_heatmap.png")

    pairs = corr.stack().reset_index()
    pairs.columns = ["var1", "var2", "r"]
    pairs = pairs[(pairs["var1"] < pairs["var2"]) & (pairs["r"].abs() >= 0.5)]
    pairs = pairs.sort_values("r", key=abs, ascending=False)
    pairs.round(3).to_csv(OUT / "corr_pairs_over_0.5.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    df["Offer"] = df["Offer"].fillna("None")
    df["Age Group"] = pd.cut(df["Age"], [0, 50, 60, 70, 80, 200], right=False,
                             labels=["0-49", "50-59", "60-69", "70-79", "80+"])

    plot_distribution(df)
    plot_reasons(df)
    contract = plot_rate(df, "Contract", "churn_by_contract.png", "Churn Rate by Contract",
                         ["Month-to-Month", "One Year", "Two Year"])
    offer = plot_rate(df, "Offer", "churn_by_offer.png", "Churn Rate by Offer",
                      ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
    age = plot_rate(df, "Age Group", "churn_by_age.png", "Churn Rate by Age Group")
    plot_corr(df)

    print("이탈률:", round(df[TARGET].eq("Yes").mean() * 100, 1))
    print("계약별:", contract.to_dict())
    print("Offer별:", offer.to_dict())
    print("연령대별:", age.to_dict())


if __name__ == "__main__":
    main()
