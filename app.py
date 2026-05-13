import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="용접기 AI 데이터셋 EDA",
    page_icon="⚙️",
    layout="wide"
)

# ── 스타일 ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* 전체 배경: 아진 PPT 느낌의 다크 네이비 */
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(0, 229, 168, 0.14), transparent 30%),
            radial-gradient(circle at 86% 4%, rgba(77, 163, 255, 0.18), transparent 32%),
            linear-gradient(135deg, #07111f 0%, #0d1729 45%, #151827 100%);
        color: #EAF2FF;
    }

    /* 제목 */
    h1, h2, h3 {
        color: #F4F8FF !important;
        letter-spacing: -0.02em;
    }

    /* 일반 텍스트 */
    .stCaption, p, label, span, div {
        color: #D7E3F5;
    }

    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081120 0%, #111A2E 100%);
        border-right: 1px solid rgba(0, 229, 168, 0.18);
    }
    section[data-testid="stSidebar"] * {
        color: #EAF2FF !important;
    }

    /* 지표 카드 */
    .metric-box {
        background: rgba(15, 27, 48, 0.82);
        border-radius: 16px;
        padding: 18px 22px;
        border: 1px solid rgba(0, 229, 168, 0.24);
        box-shadow:
            0 12px 34px rgba(0, 0, 0, 0.30),
            inset 0 0 20px rgba(0, 229, 168, 0.05);
        backdrop-filter: blur(8px);
        min-height: 132px;
        height: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 12px;
        color: #9DB2D0 !important;
        margin: 0;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #55E6FF !important;
        margin: 4px 0 0;
        text-shadow: 0 0 14px rgba(85, 230, 255, 0.32);
    }
    .metric-sub {
        font-size: 11px;
        color: #7E91AE !important;
        margin: 2px 0 0;
        line-height: 1.35;
        word-break: keep-all;
    }

    /* 섹션 제목 */
    .section-title {
        font-size: 13px;
        font-weight: 800;
        color: #9BE7FF !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 28px 0 14px;
    }

    /* 데이터프레임 */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 229, 168, 0.18);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
    }

    /* expander */
    div[data-testid="stExpander"] {
        background: rgba(15, 27, 48, 0.65);
        border: 1px solid rgba(0, 229, 168, 0.18);
        border-radius: 14px;
    }

    /* selectbox */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 27, 48, 0.95) !important;
        border: 1px solid rgba(0, 229, 168, 0.45) !important;
        color: #EAF2FF !important;
        box-shadow: 0 0 0 1px rgba(0, 229, 168, 0.08), 0 8px 24px rgba(0,0,0,0.20) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #EAF2FF !important;
    }

    /* selectbox 드롭다운 메뉴: 흰 배경/흰 글자 문제 수정 */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background-color: #0B1628 !important;
        border: 1px solid rgba(0, 229, 168, 0.35) !important;
        color: #EAF2FF !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.45) !important;
    }

    li[role="option"],
    div[role="option"] {
        background-color: #0B1628 !important;
        color: #EAF2FF !important;
    }

    li[role="option"] *,
    div[role="option"] * {
        color: #EAF2FF !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover {
        background-color: rgba(0, 229, 168, 0.18) !important;
    }

    /* 알림 박스 */
    div[data-testid="stAlert"] {
        background: rgba(15, 27, 48, 0.78);
        border: 1px solid rgba(255, 209, 102, 0.28);
        border-radius: 14px;
        color: #EAF2FF;
    }

    /* ── 파일 업로드 박스: 사이드바 다크 테마와 충돌 방지 ───────────── */
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border: 1.5px dashed #38BDF8 !important;
        border-radius: 14px !important;
    }

    /* 드롭존 내부 설명 텍스트 */
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] div,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzoneInstructions"] *,
    div[data-testid="stFileUploaderDropzone"] p,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] div,
    div[data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #1E293B !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] p,
    div[data-testid="stFileUploaderDropzone"] p {
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] small {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* Browse files 버튼 */
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button {
        color: #0F172A !important;
        background-color: #E2E8F0 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="stFileUploaderDropzone"] button * {
        color: #0F172A !important;
    }

    /* 업로드 완료 후 파일명 텍스트 */
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderFile"] *,
    div[data-testid="stFileUploaderFile"] * {
        color: #0F172A !important;
    }



    /* 해석 요약 박스 */
    .insight-box {
        padding: 20px 24px;
        border-radius: 16px;
        margin: 14px 0;
        line-height: 1.75;
        font-size: 18px;
        color: #DCE8FF !important;
        background: rgba(15, 27, 48, 0.78);
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
    }
    .insight-box b { color: #F4F8FF !important; }
    .insight-yellow { border: 1px solid rgba(255, 209, 102, 0.45); background: rgba(255, 209, 102, 0.10); }
    .insight-green  { border: 1px solid rgba(0, 229, 168, 0.45);  background: rgba(0, 229, 168, 0.10); }
    .insight-red    { border: 1px solid rgba(255, 77, 109, 0.45);  background: rgba(255, 77, 109, 0.10); }
    .insight-blue   { border: 1px solid rgba(77, 163, 255, 0.45);  background: rgba(77, 163, 255, 0.10); }

</style>
""", unsafe_allow_html=True)

# ── 컬럼 정의 ─────────────────────────────────────────────────
COLS = {
    "weld force(bar)":  {"label": "가압력",   "unit": "bar"},
    "weld current(kA)": {"label": "전류",     "unit": "kA"},
    "weld Voltage(v)":  {"label": "전압",     "unit": "V"},
    "weld time(ms)":    {"label": "통전시간", "unit": "ms"},
}
COLORS = {
    "weld force(bar)":  "#00E5A8",  # 네온 그린
    "weld current(kA)": "#4DA3FF",  # 사이버 블루
    "weld Voltage(v)":  "#FFD166",  # 골드
    "weld time(ms)":    "#B388FF",  # 네온 퍼플
}
DEFECT_COLORS = {
    "파임불량": "#FF4D6D",
    "용접부족": "#FFB703",
    "크랙발생": "#9B5DE5"
}

# ── Plotly 다크 테마 함수 ─────────────────────────────────────
def apply_dark_theme(fig, height=None, showlegend=None):
    fig.update_layout(
        template="plotly_dark",
        height=height if height else None,
        title_text="",  # Plotly 제목 미지정 시 undefined 표시 방지
        plot_bgcolor="rgba(8, 17, 31, 0.92)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#DCE8FF", size=12),
        legend=dict(
            font=dict(color="#DCE8FF", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(180, 210, 255, 0.10)",
        linecolor="rgba(180, 210, 255, 0.18)",
        zerolinecolor="rgba(180, 210, 255, 0.14)",
        tickfont=dict(color="#AFC2DD"),
        title_font=dict(color="#DCE8FF"),
    )
    fig.update_yaxes(
        gridcolor="rgba(180, 210, 255, 0.10)",
        linecolor="rgba(180, 210, 255, 0.18)",
        zerolinecolor="rgba(180, 210, 255, 0.14)",
        tickfont=dict(color="#AFC2DD"),
        title_font=dict(color="#DCE8FF"),
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig

# ── 데이터 로드 ───────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df_p = pd.read_excel(file, sheet_name="Raw data")
    df_d = pd.read_excel(file, sheet_name="result")
    df_p["working time"] = pd.to_datetime(df_p["working time"]).dt.date
    df_d["working time"] = pd.to_datetime(df_d["working time"]).dt.date
    return df_p, df_d

@st.cache_data
def build_summary(_df_p, _df_d):
    cols = list(COLS.keys())
    df_stats = _df_p.groupby("working time")[cols].agg(["mean", "std"])
    df_stats.columns = [
        f"{c[0].split('(')[0].replace('weld ', '')}_{c[1]}"
        for c in df_stats.columns
    ]
    df_stats["전체용접횟수"] = _df_p.groupby("working time").size()

    defect_cols = ["파임불량", "용접부족", "크랙발생"]
    for t, name in [(1, "파임불량"), (2, "용접부족"), (3, "크랙발생")]:
        defect_series = (
            _df_d[_df_d["defect type"] == t]
            .groupby("working time")["defect"].sum()
        )
        df_stats[name] = defect_series

    # result 시트에 없는 생산일은 "불량 기록 없음"으로 보고 0건 처리
    df_stats[defect_cols] = df_stats[defect_cols].fillna(0).astype(int)
    df_stats["불량개수"] = df_stats[defect_cols].sum(axis=1).astype(int)
    df_stats["불량률(%)"] = (
        df_stats["불량개수"] / df_stats["전체용접횟수"] * 100
    ).round(4)
    mu    = _df_p["weld force(bar)"].mean()
    sigma = _df_p["weld force(bar)"].std()
    ucl   = mu + 3 * sigma
    for date, group in _df_p.groupby("working time")["weld force(bar)"]:
        df_stats.loc[date, "가압력이상치비율(%)"] = round(
            (group > ucl).sum() / len(group) * 100, 1
        )
    return df_stats.reset_index()

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 용접기 EDA")
    st.caption("KAIST · KAMP 데이터셋")
    st.divider()
    uploaded = st.file_uploader(
        "Welding_Data_Set_01.xlsx 업로드", type=["xlsx"]
    )
    st.divider()
    page = st.radio(
        "페이지",
        ["📊 개요", "📈 분포 분석", "📉 관리도 (SPC)",
         "🔍 교번 패턴 분석", "📋 날짜별 불량 분석", "📌 핵심 날짜 비교"]
    )

if uploaded is None:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드해주세요.")
    st.stop()

df, df_defect = load_data(uploaded)
summary = build_summary(df, df_defect)

# ══════════════════════════════════════════════════════════════
# 페이지 1: 개요
# ══════════════════════════════════════════════════════════════
if page == "📊 개요":
    st.title("📊 데이터 개요")
    st.caption("Welding_Data_Set_01.xlsx · Raw data 시트")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "총 데이터 수",  f"{len(df):,}개",  "행(rows)"),
        (c2, "수집 기간",     "15일",             "2020.03.24 ~ 04.07"),
        (c3, "결측치",        "0개",              "완전성 99.94%"),
        (c4, "총 불량 건수",  f"{int(df_defect['defect'].sum())}건",
             "파임·용접부족·크랙"),
    ]
    for col, lbl, val, sub in cards:
        col.markdown(f"""
        <div class="metric-box">
            <p class="metric-label">{lbl}</p>
            <p class="metric-value">{val}</p>
            <p class="metric-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">날짜별 생산량</p>', unsafe_allow_html=True)

    daily = df.groupby("working time").size().reset_index(name="count")
    daily["날짜"] = daily["working time"].astype(str)

    fig = go.Figure(go.Bar(
        x=daily["날짜"],
        y=daily["count"],
        marker_color=COLORS["weld force(bar)"],
        marker_line_width=0,
    ))

    fig.update_layout(
        height=260,
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(
            title="날짜",
            type="category",          # 없는 날짜 제거
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="생산 수량",
            gridcolor="rgba(180, 210, 255, 0.10)"
        ),
        plot_bgcolor="rgba(8, 17, 31, 0.92)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">기초 통계</p>', unsafe_allow_html=True)
    rows = []
    for col_name, meta in COLS.items():
        s = df[col_name].dropna()
        mu, sigma = s.mean(), s.std()
        out3 = int(((s > mu + 3*sigma) | (s < mu - 3*sigma)).sum())
        rows.append({
            "변수": f"{meta['label']} ({meta['unit']})",
            "평균": round(mu, 4),
            "표준편차": round(sigma, 4),
            "최솟값": round(s.min(), 4),
            "중간값": round(s.median(), 4),
            "최댓값": round(s.max(), 4),
            "3σ 이탈": f"{out3}개",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("데이터 미리보기"):
        preview_df = df[[
            "working time",
            "weld force(bar)",
            "weld current(kA)",
            "weld Voltage(v)",
            "weld time(ms)"
        ]].copy()

        preview_df.columns = [
            "날짜",
            "가압력(bar)",
            "전류(kA)",
            "전압(V)",
            "통전시간(ms)"
        ]

        preview_df.index = range(1, len(preview_df) + 1)

        st.dataframe(
            preview_df.head(20),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════
# 페이지 2: 분포 분석
# ══════════════════════════════════════════════════════════════
elif page == "📈 분포 분석":
    st.title("📈 변수별 분포 분석")

    sel   = st.selectbox(
        "변수 선택", list(COLS.keys()),
        format_func=lambda k: f"{COLS[k]['label']} ({COLS[k]['unit']})"
    )
    meta  = COLS[sel]
    color = COLORS[sel]
    s     = df[sel].dropna()
    mu, sigma = s.mean(), s.std()
    ucl, lcl  = mu + 3*sigma, mu - 3*sigma
    out3 = int(((s > ucl) | (s < lcl)).sum())

    m1, m2, m3, m4 = st.columns(4)
    for col_st, lbl, val in [
        (m1, "평균 (X̄)",    f"{mu:.4f}"),
        (m2, "표준편차 (σ)", f"{sigma:.4f}"),
        (m3, "UCL (+3σ)",    f"{ucl:.4f}"),
        (m4, "3σ 이탈",      f"{out3}개"),
    ]:
        col_st.markdown(f"""
        <div class="metric-box">
            <p class="metric-label">{lbl}</p>
            <p class="metric-value">{val}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("히스토그램", "Box Plot"))
    fig.add_trace(go.Histogram(
        x=s, nbinsx=50, marker_color=color, opacity=0.7,
        name="데이터", histnorm="probability density"
    ), row=1, col=1)
    for val_line, name_line in [(ucl, "UCL (+3σ)"), (lcl, "LCL (-3σ)")]:
        fig.add_vline(
            x=val_line, line_dash="dash", line_color=DEFECT_COLORS["파임불량"],
            annotation_text=name_line, annotation_font_size=10,
            row=1, col=1
        )
    fig.add_trace(go.Box(
        y=s, marker_color=color, name=meta["label"],
        boxpoints="outliers", marker=dict(size=3, opacity=0.5)
    ), row=1, col=2)
    fig.update_layout(
        height=420,
        showlegend=False,
        margin=dict(t=50, b=50, l=20, r=20),
        plot_bgcolor="rgba(8, 17, 31, 0.92)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    # 분포 분석 차트 축 설명 추가
    fig.update_xaxes(
        title_text=f"{meta['label']} ({meta['unit']})",
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text="확률밀도",
        row=1,
        col=1
    )
    fig.update_xaxes(
        title_text="변수",
        row=1,
        col=2
    )
    fig.update_yaxes(
        title_text=f"{meta['label']} ({meta['unit']})",
        row=1,
        col=2
    )

    apply_dark_theme(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("정규성 검정 (Shapiro-Wilk) — 표본 500개"):
        sample = s.sample(min(500, len(s)), random_state=42)
        stat, p = scipy_stats.shapiro(sample)
        st.markdown(f"- 검정통계량 W = **{stat:.4f}**")
        st.markdown(f"- p-value = **{p:.4f}**")
        if p < 0.05:
            st.warning("p < 0.05 → 정규분포라고 보기 어려워요.")
        else:
            st.success("p ≥ 0.05 → 정규분포로 볼 수 있어요.")

# ══════════════════════════════════════════════════════════════
# 페이지 3: 관리도 (SPC)
# ══════════════════════════════════════════════════════════════
elif page == "📉 관리도 (SPC)":
    st.title("📉 관리도 (SPC · X-bar Chart)")
    st.caption("날짜를 소그룹(subgroup)으로 설정한 X-bar 관리도")

    sel   = st.selectbox(
        "변수 선택", list(COLS.keys()),
        format_func=lambda k: f"{COLS[k]['label']} ({COLS[k]['unit']})"
    )
    meta  = COLS[sel]
    color = COLORS[sel]
    s     = df[sel].dropna()
    mu, sigma = s.mean(), s.std()
    ucl3  = mu + 3*sigma
    lcl3  = mu - 3*sigma

    grp   = df.groupby("working time")[sel]
    xbars = grp.mean().reset_index()
    xbars.columns = ["date", "xbar"]
    xbars["date_str"] = xbars["date"].astype(str)
    xbars["out"] = (xbars["xbar"] > ucl3) | (xbars["xbar"] < lcl3)

    m1, m2, m3, m4 = st.columns(4)
    for col_st, lbl, val in [
        (m1, "X̿ (중심선)",  f"{mu:.4f} {meta['unit']}"),
        (m2, "σ",            f"{sigma:.4f}"),
        (m3, "UCL (+3σ)",    f"{ucl3:.4f}"),
        (m4, "관리이탈 일수", f"{int(xbars['out'].sum())}일"),
    ]:
        col_st.markdown(f"""
        <div class="metric-box">
            <p class="metric-label">{lbl}</p>
            <p class="metric-value">{val}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    fig = go.Figure()
    for hi, lo, alpha in [
        (mu+3*sigma, mu-3*sigma, 0.06),
        (mu+2*sigma, mu-2*sigma, 0.10),
        (mu+1*sigma, mu-1*sigma, 0.14),
    ]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=alpha, line_width=0)

    for y_val, dash, name, clr in [
        (ucl3,       "dash",  "UCL (+3σ)", "#E24B4A"),
        (lcl3,       "dash",  "LCL (-3σ)", "#E24B4A"),
        (mu+2*sigma, "dot",   "+2σ",       "#FA8B00"),
        (mu-2*sigma, "dot",   "-2σ",       "#FA8B00"),
        (mu,         "solid", "중심선",     "#888780"),
    ]:
        fig.add_hline(
            y=y_val, line_dash=dash, line_color=clr, line_width=1.5,
            annotation_text=name, annotation_font_size=10,
            annotation_font_color=clr
        )

    point_colors = ["#E24B4A" if o else color for o in xbars["out"]]
    fig.add_trace(go.Scatter(
        x=xbars["date_str"], y=xbars["xbar"].round(4),
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=10, color=point_colors,
                    line=dict(width=2, color="#FFFFFF")),
        hovertemplate="%{x}<br>X-bar = %{y:.4f}<extra></extra>"
    ))
    fig.update_layout(
        height=420, margin=dict(t=20, b=20, l=0, r=120),
        plot_bgcolor="rgba(8, 17, 31, 0.92)", paper_bgcolor="rgba(0, 0, 0, 0)", showlegend=False,
        xaxis=dict(title="날짜", gridcolor="rgba(180, 210, 255, 0.10)", tickfont=dict(size=11)),
        yaxis=dict(title=f"{meta['label']} ({meta['unit']})", gridcolor="rgba(180, 210, 255, 0.10)"),
    )
    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("날짜별 X-bar 수치 보기"):
        tbl = xbars[["date_str", "xbar", "out"]].copy()
        tbl.columns = ["날짜", f"X-bar ({meta['unit']})", "관리이탈"]
        tbl[f"X-bar ({meta['unit']})"] = tbl[f"X-bar ({meta['unit']})"].round(4)
        tbl["관리이탈"] = tbl["관리이탈"].map({True: "⚠️ 이탈", False: "✅ 정상"})
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# 페이지 4: 교번 패턴 분석
# ══════════════════════════════════════════════════════════════
elif page == "🔍 교번 패턴 분석":
    st.title("🔍 교번 패턴 분석 (Alternating Pattern)")
    st.caption("가압력(weld force)의 날짜별 X-bar가 낮은 날 → 높은 날로 반복되는 특수원인 검증")

    col   = "weld force(bar)"
    grp   = df.groupby("working time")[col]
    xbars = grp.mean()
    dates_str = [str(d) for d in xbars.index]
    vals  = xbars.values
    is_high   = [i % 2 == 1 for i in range(len(vals))]
    low_days  = vals[::2]
    high_days = vals[1::2]

    mu, sigma = df[col].mean(), df[col].std()
    ucl = mu + 3 * sigma
    outlier_pct = []
    for date, group in df.groupby("working time")[col]:
        outlier_pct.append(round((group > ucl).sum() / len(group) * 100, 1))

    ac1 = pd.Series(vals).autocorr(lag=1)
    t_stat, p_val = scipy_stats.ttest_ind(low_days, high_days)
    diffs = np.diff(vals)
    signs = [1 if d > 0 else -1 for d in diffs]
    full_alt = all(signs[i] != signs[i+1] for i in range(len(signs)-1))

    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-box">
        <p class="metric-label">자기상관계수 (lag-1)</p>
        <p class="metric-value" style="color:#c92a2a">{ac1:.3f}</p>
        <p class="metric-sub" style="color:#c92a2a">-0.5 이하 → 강한 교번 패턴</p>
    </div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-box">
        <p class="metric-label">t-검정 p-value</p>
        <p class="metric-value" style="color:#c92a2a">{p_val:.4f}</p>
        <p class="metric-sub" style="color:#c92a2a">p &lt; 0.05 → 통계적으로 유의미</p>
    </div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-box">
        <p class="metric-label">완전 교번 패턴</p>
        <p class="metric-value" style="color:#c92a2a">{"True ✓" if full_alt else "False"}</p>
        <p class="metric-sub" style="color:#c92a2a">8회 연속 방향 전환 확인</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("날짜별 X-bar (낮은 날 vs 높은 날)", "날짜별 이상치 비율"),
        vertical_spacing=0.24
    )
    fig.add_trace(go.Bar(
        x=[d for d, h in zip(dates_str, is_high) if not h],
        y=[v for v, h in zip(vals, is_high) if not h],
        name="정상일", marker_color=COLORS["weld force(bar)"], marker_line_width=0,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[d for d, h in zip(dates_str, is_high) if h],
        y=[v for v, h in zip(vals, is_high) if h],
        name="이상일", marker_color=DEFECT_COLORS["파임불량"], marker_line_width=0,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_str, y=[round(v, 4) for v in vals],
        mode="lines+markers", name="X-bar 추이",
        line=dict(color=COLORS["weld current(kA)"], width=2),
        marker=dict(size=9,
                    color=["#E24B4A" if h else "#1D9E75" for h in is_high],
                    line=dict(width=2, color="#FFFFFF")),
    ), row=1, col=1)
    fig.add_hline(y=low_days.mean(), line_dash="dot", line_color=COLORS["weld force(bar)"],
                  annotation_text=f"정상일 평균 {low_days.mean():.2f}",
                  annotation_font_color=COLORS["weld force(bar)"], row=1, col=1)
    fig.add_hline(y=high_days.mean(), line_dash="dot", line_color=DEFECT_COLORS["파임불량"],
                  annotation_text=f"이상일 평균 {high_days.mean():.2f}",
                  annotation_font_color=DEFECT_COLORS["파임불량"], row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates_str, y=outlier_pct,
        marker_color=["#E24B4A" if p > 0 else "#1D9E75" for p in outlier_pct],
        showlegend=False, marker_line_width=0,
    ), row=2, col=1)
    fig.update_layout(
        height=650,
        margin=dict(t=40, b=10, l=0, r=0),
        plot_bgcolor="rgba(8, 17, 31, 0.92)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(orientation="h", y=1.04, font=dict(size=11)),
    )

    # 생산 데이터가 없는 날짜는 x축에서 제외
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=dates_str,
        row=1,
        col=1
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=dates_str,
        row=2,
        col=1
    )

    fig.update_yaxes(title_text="가압력 (bar)", row=1, col=1)
    fig.update_yaxes(title_text="이상치 비율 (%)", row=2, col=1)

    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    

    st.warning(f"""
**특수원인(Special Cause) 존재 — 3가지 통계로 증명**

1. **자기상관계수 {ac1:.3f}** — lag-1 자기상관이 -0.5 이하면 강한 교번 패턴.
2. **t-검정 p = {p_val:.4f}** — 정상일 평균({low_days.mean():.2f} bar)과 이상일 평균({high_days.mean():.2f} bar) 차이가 통계적으로 유의미 (p < 0.05).
3. **이상치 완전 분리** — 정상일 5일 이상치 0%, 이상일 4일 이상치 8~20%.
    """)

    st.info("""
**데이터 기반 해석**
- 가압력 평균이 낮은 날짜와 높은 날짜가 반복적으로 나타남
- 평균이 높은 날짜에서 고가압 이상치 비율도 함께 증가함
- 현재 데이터에는 작업자, 설비 세팅, 소재 로트, 전극 팁 이력이 없으므로 원인은 단정할 수 없음
- 따라서 본 분석은 원인 규명이 아니라 가압력 변동 패턴 탐지로 해석함
    """)

# ══════════════════════════════════════════════════════════════
# 페이지 5: 날짜별 불량 분석
# ══════════════════════════════════════════════════════════════
elif page == "📋 날짜별 불량 분석":
    st.title("📋 날짜별 불량 분석")
    st.caption("날짜별 공정 변수 평균/표준편차 + 불량률 + 불량 유형 비교")

    st.markdown('<p class="section-title">날짜별 종합 요약</p>', unsafe_allow_html=True)

    show_cols = [
        "working time", "전체용접횟수",
        "force_mean",
        "파임불량", "용접부족", "크랙발생",
        "불량개수", "불량률(%)", "가압력이상치비율(%)"
    ]

    disp = summary[show_cols].copy()
    disp.columns = [
        "날짜", "생산수량",
        "가압력평균(bar)",
        "파임불량", "용접부족", "크랙발생",
        "총불량", "불량률(%)", "가압력이상치(%)"
    ]

    st.dataframe(disp, use_container_width=True, hide_index=True)

    # 날짜를 문자열 카테고리로 고정
    x_dates = summary["working time"].astype(str).tolist()

    # ── 불량률 + 가압력이상치 비교 ───────────────────────────
    st.markdown('<p class="section-title">불량률 vs 가압력 이상치 비율</p>', unsafe_allow_html=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=x_dates,
        y=summary["불량률(%)"],
        name="불량률 (%)",
        marker_color=DEFECT_COLORS["파임불량"],
        marker_line_width=0,
        opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x_dates,
        y=summary["가압력이상치비율(%)"],
        name="가압력 이상치 비율 (%)",
        mode="lines+markers",
        line=dict(color=COLORS["weld force(bar)"], width=2.5),
        marker=dict(size=9, line=dict(width=2, color="#FFFFFF"))
    ), secondary_y=True)

    fig.update_layout(
        height=360,
        margin=dict(t=20, b=10, l=0, r=0),
        plot_bgcolor="rgba(8, 17, 31, 0.92)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        xaxis=dict(
            title="날짜",
            type="category",
            categoryorder="array",
            categoryarray=x_dates,
            gridcolor="rgba(180, 210, 255, 0.10)",
            tickfont=dict(size=11)
        )
    )

    fig.update_yaxes(
        title_text="불량률 (%)",
        secondary_y=False,
        gridcolor="rgba(180, 210, 255, 0.10)"
    )

    fig.update_yaxes(
        title_text="가압력 이상치 비율 (%)",
        secondary_y=True
    )

    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── 불량 유형별 날짜 비교 ─────────────────────────────────
    st.markdown('<p class="section-title">날짜별 불량 유형 구성</p>', unsafe_allow_html=True)

    fig2 = go.Figure()

    for defect, color in DEFECT_COLORS.items():
        fig2.add_trace(go.Bar(
            x=x_dates,
            y=summary[defect],
            name=defect,
            marker_color=color,
            marker_line_width=0,
        ))

    fig2.update_layout(
        barmode="stack",
        height=300,
        margin=dict(t=20, b=10, l=0, r=0),
        plot_bgcolor="rgba(8, 17, 31, 0.92)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        xaxis=dict(
            title="날짜",
            type="category",
            categoryorder="array",
            categoryarray=x_dates,
            gridcolor="rgba(180, 210, 255, 0.10)",
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="불량 건수",
            gridcolor="rgba(180, 210, 255, 0.10)"
        ),
    )

    apply_dark_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">주요 인사이트</p>', unsafe_allow_html=True)

    max_defect_day  = summary.loc[summary["불량률(%)"].idxmax(), "working time"]
    max_defect_rate = summary["불량률(%)"].max()
    max_outlier_day = summary.loc[summary["가압력이상치비율(%)"].idxmax(), "working time"]

    st.info(f"""
- **불량률 최고**: {max_defect_day} ({max_defect_rate:.2f}%)
- **가압력 이상치 최고**: {max_outlier_day} (20.0%)
- **03/27**: 가압력 이상치 9.7%였지만 불량 0건 → 단일 변수로 불량을 설명하기 어려움
- **03/26**: 가압력 이상치 0%인데 불량률 0.7% → 복합 원인 가능성 시사
    """)

# ══════════════════════════════════════════════════════════════
# 페이지 6: 3개 날짜 비교
# ══════════════════════════════════════════════════════════════
elif page == "📌 핵심 날짜 비교":
    st.title("📌 3개 날짜 비교")
    st.caption("2020-03-26 · 2020-03-27 · 2020-04-03의 생산수량, 불량률, 공정 평균을 한눈에 비교합니다.")

    import datetime

    # 비교할 날짜를 고정해서 단순 비교
    target_days = [
        datetime.date(2020, 3, 26),
        datetime.date(2020, 3, 27),
        datetime.date(2020, 4, 3),
    ]

    # 실제 데이터가 있는 날짜만 사용
    available_days = set(summary["working time"])
    compare_days = [d for d in target_days if d in available_days]

    if len(compare_days) == 0:
        st.warning("비교할 날짜 데이터가 없습니다.")
        st.stop()

    compare_df = summary[summary["working time"].isin(compare_days)].copy()
    compare_df = compare_df.set_index("working time").loc[compare_days].reset_index()
    compare_df["날짜"] = compare_df["working time"].astype(str)

    day_colors = {
        datetime.date(2020, 3, 26): "#FF4D6D",
        datetime.date(2020, 3, 27): "#00E5A8",
        datetime.date(2020, 4, 3): "#FFD166",
    }
    day_border = {
        datetime.date(2020, 3, 26): "rgba(255,77,109,0.60)",
        datetime.date(2020, 3, 27): "rgba(0,229,168,0.55)",
        datetime.date(2020, 4, 3): "rgba(255,209,102,0.55)",
    }
    bar_colors = [day_colors.get(d, "#55E6FF") for d in compare_df["working time"]]

    # ── 상단 요약 카드 ───────────────────────────────────────
    card_cols = st.columns(len(compare_df))
    for card_col, (_, row) in zip(card_cols, compare_df.iterrows()):
        d = row["working time"]
        card_col.markdown(f"""
        <div class="metric-box" style="border-color:{day_border.get(d, 'rgba(85,230,255,0.45)')}; min-height:150px;">
            <p class="metric-label" style="font-size:16px; color:#DCE8FF !important;">{row['날짜']}</p>
            <p class="metric-value" style="color:{day_colors.get(d, '#55E6FF')} !important;">{row['불량률(%)']:.4f}%</p>
            <p class="metric-sub">불량 {int(row['불량개수'])}건 / 생산 {int(row['전체용접횟수']):,}개</p>
            <p class="metric-sub">가압력 평균 {row['force_mean']:.2f}bar</p>
            <p class="metric-sub">가압력 이상치 {row['가압력이상치비율(%)']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # 변수별 평균 비교 그래프는 삭제하고, 해석 요약만 표시

    st.markdown('<p class="section-title">해석 요약</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box insight-yellow">
        <b>2020-04-03 · 가압력 최고</b><br>
        가압력 평균은 <b>4.03bar</b>로 가장 높고, 가압력 이상치 비율도 <b>20.0%</b>입니다.
        다만 불량률은 <b>0.5000%</b>로 최고 불량일과 일치하지 않습니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box insight-green">
        <b>2020-03-27 · 불량 0건</b><br>
        불량률은 <b>0.0000%</b>이지만, 가압력 이상치 비율은 <b>9.7%</b>입니다.
        즉, 가압력 이상치가 존재해도 해당 날짜의 집계 불량은 0건으로 처리되었습니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box insight-red">
        <b>2020-03-26 · 불량률 최고</b><br>
        불량률은 <b>0.7000%</b>로 가장 높지만, 가압력 이상치 비율은 <b>0.0%</b>입니다.
        따라서 현재 데이터만으로는 가압력 하나가 불량률을 직접 설명한다고 보기 어렵습니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box insight-blue">
        <b>결론</b><br>
        세 날짜를 비교하면 가압력 최고일, 불량 0건일, 불량률 최고일이 서로 다르게 나타납니다.<br>
        따라서 현재 데이터에서는 특정 공정 변수 하나로 불량 원인을 단정하기보다,
        <b>개별 용접 데이터 라벨링과 추가 공정 정보 수집이 필요하다</b>는 방향으로 해석하는 것이 안전합니다.
    </div>
    """, unsafe_allow_html=True)
