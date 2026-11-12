import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="청소년 우울감 분석 및 예측", page_icon="📊", layout="wide")

st.title("📊 청소년 우울감 경험률 분석 및 미래 예측")
st.caption("질병관리청 청소년건강행태조사 자료를 활용한 통계 분석 및 선형회귀 실습")

st.info(
    "이 웹앱은 설문에서 보고된 '우울감 경험률'을 분석합니다. "
    "의학적 우울증을 진단하거나 개인의 정신건강을 판정하는 도구가 아닙니다."
)

# 공식 발표자료에서 확인한 2016~2025년 성별 우울감 경험률
# School_Level과 Grade는 성별 전체 자료이므로 '전체'로 표시합니다.
sample = pd.DataFrame([
    [2016, "남학생", "전체", "전체", 20.9],
    [2017, "남학생", "전체", "전체", 20.3],
    [2018, "남학생", "전체", "전체", 21.1],
    [2019, "남학생", "전체", "전체", 22.2],
    [2020, "남학생", "전체", "전체", 20.1],
    [2021, "남학생", "전체", "전체", 22.4],
    [2022, "남학생", "전체", "전체", 24.2],
    [2023, "남학생", "전체", "전체", 21.4],
    [2024, "남학생", "전체", "전체", 23.1],
    [2025, "남학생", "전체", "전체", 21.7],
    [2016, "여학생", "전체", "전체", 30.5],
    [2017, "여학생", "전체", "전체", 30.3],
    [2018, "여학생", "전체", "전체", 33.6],
    [2019, "여학생", "전체", "전체", 34.6],
    [2020, "여학생", "전체", "전체", 30.7],
    [2021, "여학생", "전체", "전체", 31.4],
    [2022, "여학생", "전체", "전체", 33.5],
    [2023, "여학생", "전체", "전체", 30.9],
    [2024, "여학생", "전체", "전체", 32.5],
    [2025, "여학생", "전체", "전체", 29.9],
], columns=["Year", "Gender", "School_Level", "Grade", "Depression_Rate"])

REQUIRED = ["Year", "Gender", "School_Level", "Grade", "Depression_Rate"]

with st.sidebar:
    st.header("데이터 설정")
    uploaded = st.file_uploader("CSV 데이터 업로드", type=["csv"])
    st.caption("CSV 필수 컬럼: Year, Gender, School_Level, Grade, Depression_Rate")
    st.caption("업로드하지 않으면 앱에 포함된 2016~2025년 공식 성별 통계 예시를 사용합니다.")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error("CSV에 다음 컬럼이 없습니다: " + ", ".join(missing))
            st.stop()
        st.success("업로드한 CSV를 불러왔습니다.")
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    df = sample.copy()
    st.warning(
        "현재 기본 데이터는 성별 전체 통계입니다. 학교급·학년별 비교를 하려면 "
        "질병관리청 통계집에서 해당 세부 통계를 확인해 같은 형식의 CSV로 업로드하세요."
    )

# 데이터 정리
df = df[REQUIRED].copy()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Depression_Rate"] = pd.to_numeric(df["Depression_Rate"], errors="coerce")
df = df.dropna(subset=["Year", "Depression_Rate"])
df["Year"] = df["Year"].astype(int)
df["Gender"] = df["Gender"].fillna("전체").astype(str)
df["School_Level"] = df["School_Level"].fillna("전체").astype(str)
df["Grade"] = df["Grade"].fillna("전체").astype(str)
df = df[df["Depression_Rate"].between(0, 100)]

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다. 연도와 우울감 경험률을 확인하세요.")
    st.stop()

st.subheader("1. 데이터 살펴보기")
c1, c2, c3 = st.columns(3)
c1.metric("데이터 행 수", f"{len(df):,}개")
c2.metric("조사 연도 범위", f"{df['Year'].min()}~{df['Year'].max()}년")
c3.metric("평균 우울감 경험률", f"{df['Depression_Rate'].mean():.1f}%")
with st.expander("수집한 데이터 표 보기"):
    st.dataframe(df.sort_values(["Year", "Gender", "School_Level", "Grade"]), use_container_width=True)
    st.download_button(
        "현재 데이터 CSV 다운로드",
        df.to_csv(index=False).encode("utf-8-sig"),
        "youth_depression_data.csv",
        "text/csv"
    )

st.subheader("2. 연도별 우울감 경험률 변화")
genders = sorted(df["Gender"].unique().tolist())
selected_genders = st.multiselect("성별 선택", genders, default=genders)
trend = df[df["Gender"].isin(selected_genders)].copy()
if not trend.empty:
    # 학교급·학년별 데이터가 함께 있을 경우 같은 연도/성별 중복을 피하려고
    # 현재 표에서는 연도와 성별별 평균을 표시합니다.
    trend_summary = trend.groupby(["Year", "Gender"], as_index=False)["Depression_Rate"].mean()
    fig = px.line(
        trend_summary, x="Year", y="Depression_Rate", color="Gender",
        markers=True, title="연도별 우울감 경험률",
        labels={"Year": "조사 연도", "Depression_Rate": "우울감 경험률(%)", "Gender": "성별"}
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("성별을 하나 이상 선택하세요.")

st.subheader("3. 학교급·학년별 비교")
col_a, col_b = st.columns(2)
with col_a:
    school_summary = df[df["School_Level"] != "전체"].groupby(
        ["School_Level"], as_index=False
    )["Depression_Rate"].mean()
    if not school_summary.empty:
        f = px.bar(school_summary, x="School_Level", y="Depression_Rate",
                   title="학교급별 평균 우울감 경험률",
                   labels={"School_Level": "학교급", "Depression_Rate": "우울감 경험률(%)"})
        st.plotly_chart(f, use_container_width=True)
    else:
        st.info("현재 데이터에는 학교급별 세부 값이 없습니다. 학교급별 통계를 포함한 CSV를 업로드하세요.")
with col_b:
    grade_summary = df[df["Grade"] != "전체"].groupby(
        ["Grade"], as_index=False
    )["Depression_Rate"].mean()
    if not grade_summary.empty:
        f = px.bar(grade_summary, x="Grade", y="Depression_Rate",
                   title="학년별 평균 우울감 경험률",
                   labels={"Grade": "학년", "Depression_Rate": "우울감 경험률(%)"})
        st.plotly_chart(f, use_container_width=True)
    else:
        st.info("현재 데이터에는 학년별 세부 값이 없습니다. 학년별 통계를 포함한 CSV를 업로드하세요.")

st.subheader("4. 선형회귀 모델로 미래 경험률 예측")
st.write("독립 변수(X)는 조사 연도(Year), 종속 변수(y)는 우울감 경험률(Depression_Rate)입니다.")

gender_options = sorted(df["Gender"].unique().tolist())
school_options = sorted(df["School_Level"].unique().tolist())
grade_options = sorted(df["Grade"].unique().tolist())
p1, p2, p3 = st.columns(3)
with p1:
    model_gender = st.selectbox("모델에 사용할 성별", gender_options)
with p2:
    model_school = st.selectbox("모델에 사용할 학교급", school_options)
with p3:
    model_grade = st.selectbox("모델에 사용할 학년", grade_options)

model_df = df[
    (df["Gender"] == model_gender) &
    (df["School_Level"] == model_school) &
    (df["Grade"] == model_grade)
].groupby("Year", as_index=False)["Depression_Rate"].mean().sort_values("Year")

if len(model_df) < 5:
    st.error("선택한 조건의 데이터가 5개 연도 미만입니다. 최소 5개 연도 이상의 자료를 준비하세요.")
else:
    X = model_df[["Year"]]
    y = model_df["Depression_Rate"]
    split = max(3, int(len(model_df) * 0.7))
    if split >= len(model_df):
        split = len(model_df) - 1
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred) if len(y_test) >= 2 else np.nan

    m1, m2 = st.columns(2)
    m1.metric("MAE (평균 절대 오차)", f"{mae:.2f}%p")
    m2.metric("R² (결정계수)", "계산 불가" if np.isnan(r2) else f"{r2:.3f}")
    st.caption(
        f"시간 순서대로 앞의 {split}개 연도를 학습에 사용하고, "
        f"뒤의 {len(model_df)-split}개 연도를 평가에 사용했습니다."
    )

    future_year = st.slider(
        "예측할 연도", min_value=int(model_df["Year"].max()) + 1,
        max_value=int(model_df["Year"].max()) + 10,
        value=int(model_df["Year"].max()) + 5
    )
    future_value = float(model.predict(pd.DataFrame({"Year": [future_year]}))[0])
    st.metric(f"{future_year}년 선형회귀 예측값", f"{future_value:.1f}%")
    st.caption("예측값은 과거 추세를 직선으로 연장한 통계적 추정치이며 실제 미래 결과를 보장하지 않습니다.")

    all_years = pd.DataFrame({"Year": np.arange(int(model_df["Year"].min()), future_year + 1)})
    all_years["Predicted"] = model.predict(all_years[["Year"]])
    chart = px.scatter(model_df, x="Year", y="Depression_Rate",
                       title="실제 데이터와 선형회귀 예측선",
                       labels={"Year": "조사 연도", "Depression_Rate": "우울감 경험률(%)"})
    chart.add_scatter(x=all_years["Year"], y=all_years["Predicted"],
                      mode="lines", name="선형회귀 예측선")
    chart.add_scatter(x=X_test["Year"], y=test_pred, mode="markers",
                      name="평가용 예측값")
    chart.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(chart, use_container_width=True)

    comparison = pd.DataFrame({
        "Year": X_test["Year"].values,
        "실제 경험률(%)": y_test.values,
        "예측 경험률(%)": np.round(test_pred, 2)
    })
    st.write("평가용 데이터의 실제값과 예측값")
    st.dataframe(comparison, use_container_width=True)

st.divider()
st.caption(
    "자료 출처: 질병관리청 청소년건강행태조사(https://www.kdca.go.kr/yhs/). "
    "기본 예시의 성별 수치는 2016~2025년 공식 발표 자료를 바탕으로 입력했습니다. "
    "세부 학교급·학년 자료는 원본 통계집을 확인해 추가해야 합니다."
)
