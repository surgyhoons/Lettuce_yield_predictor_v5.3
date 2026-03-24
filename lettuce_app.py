"""
식물공장 상추 수확량 예측 시스템 — Streamlit 앱
실행: streamlit run lettuce_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import os, io

# ──────────────────────────────────────────────────────────────
# 0. 페이지 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="상추 수확량 예측",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# 1. 고정 설정 (변경 불필요)
# ──────────────────────────────────────────────────────────────
FIXED_BED_CONFIG = {
    1:40, 2:40, 3:40, 4:40, 5:40, 6:40, 7:40,
    8:32, 9:32, 10:32, 11:32, 12:32, 13:32, 14:32,
    15:32, 16:32, 17:32, 18:32,
    19:40, 20:40,
}
PLANTS_PER_TRAY   = 16
PLANTS_PER_GUTTER = 13

DB_COLS = [
    "batch_id", "sow_date", "transplant_date", "plant_date",
    "harvest_date", "grow_days", "bed_type", "bed_id",
    "tray_or_gutter", "weight_per_plant_g", "loss_rate",
    "actual_yield", "actual_weight_kg", "note",
]

WEEKDAYS_KR = ["월", "화", "수", "목", "금", "토", "일"]

# ──────────────────────────────────────────────────────────────
# 2. DB 로드 / 저장 (세션 + CSV)
# ──────────────────────────────────────────────────────────────
DB_FILE = "DB_배치데이터.csv"

def load_db_from_file(uploaded=None):
    """업로드된 파일 또는 로컬 파일에서 DB 로드"""
    if uploaded is not None:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
    elif os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE, encoding="utf-8-sig")
    else:
        df = pd.DataFrame(columns=DB_COLS)
    for c in DB_COLS:
        if c not in df.columns:
            df[c] = None
    return df[DB_COLS].copy()

def save_db(df: pd.DataFrame):
    df.to_csv(DB_FILE, index=False, encoding="utf-8-sig")

def get_db() -> pd.DataFrame:
    if "db" not in st.session_state:
        st.session_state["db"] = load_db_from_file()
    return st.session_state["db"]

def set_db(df: pd.DataFrame):
    st.session_state["db"] = df
    save_db(df)

# ──────────────────────────────────────────────────────────────
# 3. 예측 계산 함수
# ──────────────────────────────────────────────────────────────
def prepare_db(df: pd.DataFrame, loss_rate: float,
               default_weight_g: float, mgs_weight_g: float) -> pd.DataFrame:
    d = df.copy()
    for col in ["sow_date", "transplant_date", "plant_date", "harvest_date"]:
        d[col] = pd.to_datetime(d[col], errors="coerce")
    d["loss_rate"]          = pd.to_numeric(d["loss_rate"],          errors="coerce").fillna(loss_rate)
    d["tray_or_gutter"]     = pd.to_numeric(d["tray_or_gutter"],     errors="coerce")
    d["weight_per_plant_g"] = pd.to_numeric(d["weight_per_plant_g"], errors="coerce")
    d["actual_yield"]       = pd.to_numeric(d["actual_yield"],       errors="coerce")
    d["actual_weight_kg"]   = pd.to_numeric(d["actual_weight_kg"],   errors="coerce")
    # 주당 무게 기본값: MGS / 고정 별도 적용
    d.loc[d["bed_type"] == "fixed", "weight_per_plant_g"] = \
        d.loc[d["bed_type"] == "fixed", "weight_per_plant_g"].fillna(default_weight_g)
    d.loc[d["bed_type"] == "mgs",   "weight_per_plant_g"] = \
        d.loc[d["bed_type"] == "mgs",   "weight_per_plant_g"].fillna(mgs_weight_g)
    # 날짜 역전 제거
    bad = d["harvest_date"].notna() & d["plant_date"].notna() & (d["harvest_date"] < d["plant_date"])
    if bad.any():
        st.warning(f"날짜 역전 {bad.sum()}건 제외됨")
        d = d[~bad]
    # 총 재배일수 (파종 → 수확)
    d["total_days"] = (d["harvest_date"] - d["sow_date"]).dt.days
    return d

def calc_prediction(row):
    tg = row["tray_or_gutter"]
    if pd.isna(tg):
        return None, None
    ppu = PLANTS_PER_TRAY if row["bed_type"] == "fixed" else PLANTS_PER_GUTTER
    p   = round(float(tg) * ppu * (1 - row["loss_rate"]))
    k   = round(p * float(row["weight_per_plant_g"]) / 1000, 2)
    return p, k

def add_predictions(d: pd.DataFrame) -> pd.DataFrame:
    results = d.apply(calc_prediction, axis=1, result_type="expand")
    results.columns = ["predicted_plants", "predicted_kg"]
    d = d.copy()
    d["predicted_plants"] = results["predicted_plants"]
    d["predicted_kg"]     = results["predicted_kg"]
    # 실제 주당무게
    d["actual_wpg"] = None
    mask = d["actual_yield"].notna() & d["actual_weight_kg"].notna() & (d["actual_yield"] > 0)
    d.loc[mask, "actual_wpg"] = (
        d.loc[mask, "actual_weight_kg"] * 1000 / d.loc[mask, "actual_yield"]
    ).round(1)
    return d

# ──────────────────────────────────────────────────────────────
# 4. UI 헬퍼
# ──────────────────────────────────────────────────────────────
def fmt_d(v, fmt="%m-%d"):
    try:    return pd.Timestamp(v).strftime(fmt)
    except: return "—"

def diff_str(actual, pred):
    if actual is None or pred is None or pd.isna(actual) or pd.isna(pred):
        return ""
    d = round(float(actual) - float(pred), 1)
    return f"+{d}" if d >= 0 else str(d)

def metric_card(label, value, sub=None, color=None):
    """작은 지표 카드 HTML"""
    color_style = f"color:{color};" if color else ""
    sub_html    = f"<div style='font-size:11px;color:#888;margin-top:2px'>{sub}</div>" if sub else ""
    return (
        f"<div style='background:#f8f8f8;border-radius:8px;padding:12px 16px;text-align:center'>"
        f"<div style='font-size:11px;color:#888;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:20px;font-weight:600;{color_style}'>{value}</div>"
        f"{sub_html}</div>"
    )

# ──────────────────────────────────────────────────────────────
# 5. 사이드바 설정
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 수확량 예측")
    st.markdown("---")

    st.subheader("⚙️ 예측 설정")
    pred_date = st.date_input("예측 기준일", value=date.today(), key="sb_pred_date")
    loss_pct  = st.slider("로스율 (%)", 0, 50, 20, step=1, key="sb_loss_pct",
                          help="기본 20% = 수확률 80%")
    loss_rate = loss_pct / 100

    st.markdown("---")
    st.subheader("주당 기본 무게 (g)")
    default_weight_g = st.number_input("고정 재배대", value=100, min_value=1, step=1, key="sb_fixed_w",
                                        help="배치별 입력값 없을 때 적용")
    mgs_weight_g     = st.number_input("MGS (NFT)", value=100, min_value=1, step=1, key="sb_mgs_w",
                                        help="MGS 배치 기본 무게 — 고정 재배대와 별도 설정 가능")

    st.markdown("---")
    st.subheader("📁 DB 파일")
    uploaded = st.file_uploader("CSV 업로드", type=["csv"], key="sb_uploader",
                                 help="DB_배치데이터.csv 형식")
    if uploaded:
        set_db(load_db_from_file(uploaded))
        st.success("DB 로드 완료")

    if st.button("💾 DB 다운로드", key="sb_dl_btn"):
        db_now = get_db()
        csv_bytes = db_now.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇ CSV 저장", csv_bytes, "DB_배치데이터.csv", "text/csv", key="sb_dl_csv")

    st.markdown("---")
    total_cap = sum(t * PLANTS_PER_TRAY for t in FIXED_BED_CONFIG.values())
    st.caption(f"고정 재배대 최대 용량: **{total_cap:,}주**")
    st.caption(f"수확률: **{100-loss_pct}%** (로스율 {loss_pct}%)")

# ──────────────────────────────────────────────────────────────
# 6. 탭 레이아웃
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 3일 대시보드", "📅 달별 전체 뷰", "📋 배치 DB 관리", "✍️ 실적 입력", "🔧 참고 용량표"]
)

# ──────────────────────────────────────────────────────────────
# TAB 1: 3일 대시보드
# ──────────────────────────────────────────────────────────────
with tab1:
    db_raw = get_db()
    if db_raw.empty:
        st.info("배치 DB가 비어 있습니다. [배치 DB 관리] 탭에서 배치를 추가하세요.")
    else:
        db = prepare_db(db_raw, loss_rate, default_weight_g, mgs_weight_g)
        db = add_predictions(db)

        dash_dates  = [pred_date, pred_date + timedelta(3), pred_date + timedelta(4)]
        target_dates = dash_dates[1:]
        target = db[db["harvest_date"].dt.date.isin(target_dates)].copy()

        def day_totals(dt):
            sub = db[db["harvest_date"].dt.date == dt]
            pp = sub["predicted_plants"].sum(skipna=True)
            pk = sub["predicted_kg"].sum(skipna=True)
            ap = sub["actual_yield"].sum(skipna=True)   if sub["actual_yield"].notna().any()       else None
            ak = sub["actual_weight_kg"].sum(skipna=True) if sub["actual_weight_kg"].notna().any() else None
            return (int(pp) if pp else 0, round(float(pk),1) if pk else 0.0,
                    int(ap) if ap else None, round(float(ak),1) if ak else None)

        d0_pp, d0_pk, d0_ap, d0_ak = day_totals(dash_dates[0])
        d3_pp, d3_pk, d3_ap, d3_ak = day_totals(dash_dates[1])
        d4_pp, d4_pk, d4_ap, d4_ak = day_totals(dash_dates[2])
        total_pp = d3_pp + d4_pp
        total_pk = round(d3_pk + d4_pk, 1)

        # ── 3일 카드 ──
        wd = lambda d: WEEKDAYS_KR[d.weekday()]
        cols = st.columns(3)

        def day_col(col, tag, dt, pp, pk, ap, ak, border_color=None):
            label = f"{dt.month}월 {dt.day}일 ({wd(dt)})"
            with col:
                if border_color:
                    st.markdown(
                        f"<div style='border-left:4px solid {border_color};padding-left:10px;margin-bottom:4px'>"
                        f"<span style='font-size:10px;font-weight:700;color:{border_color};letter-spacing:.7px'>{tag}</span>"
                        f"<br><span style='font-size:14px;font-weight:600'>{label}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='padding-left:10px;margin-bottom:4px'>"
                        f"<span style='font-size:10px;font-weight:700;color:#aaa;letter-spacing:.7px'>{tag}</span>"
                        f"<br><span style='font-size:14px;font-weight:600'>{label}</span></div>",
                        unsafe_allow_html=True,
                    )
                m1, m2 = st.columns(2)
                m1.metric("예측 주수", f"{pp:,}주" if pp else "수확 없음")
                m2.metric("예측 무게", f"{pk:.1f} kg" if pk else "—")
                if ap is not None or ak is not None:
                    st.caption("▸ 실적")
                    r1, r2 = st.columns(2)
                    r1.metric("실제 주수", f"{ap:,}주" if ap else "—",
                              delta=diff_str(ap, pp) + "주" if ap else None)
                    r2.metric("실제 무게", f"{ak:.1f} kg" if ak else "—",
                              delta=diff_str(ak, pk) + " kg" if ak else None)
                st.markdown("")

        day_col(cols[0], "D+0 · 오늘",      dash_dates[0], d0_pp, d0_pk, d0_ap, d0_ak)
        day_col(cols[1], "D+3 · 수확 예정", dash_dates[1], d3_pp, d3_pk, d3_ap, d3_ak, "#3B6D11")
        day_col(cols[2], "D+4 · 수확 예정", dash_dates[2], d4_pp, d4_pk, d4_ap, d4_ak, "#185FA5")

        # ── 합계 바 ──
        st.markdown("---")
        sc = st.columns(4)
        sc[0].metric("이번 주 예측 주수 (D+3~4)", f"{total_pp:,}주")
        sc[1].metric("이번 주 예측 무게 (D+3~4)", f"{total_pk:.1f} kg")
        valid = target[target["predicted_plants"].notna()]
        sc[2].metric("고정 재배대",
                     f"{int(valid[valid['bed_type']=='fixed']['predicted_plants'].sum()):,}주  "
                     f"/ {round(float(valid[valid['bed_type']=='fixed']['predicted_kg'].sum()),1)} kg")
        mgs_p = valid[valid["bed_type"]=="mgs"]["predicted_plants"].sum()
        mgs_k = valid[valid["bed_type"]=="mgs"]["predicted_kg"].sum()
        sc[3].metric("MGS",
                     f"{int(mgs_p):,}주 / {round(float(mgs_k),1)} kg" if mgs_p else "N/A")

        # ── 배치 상세 테이블 ──
        st.markdown("---")
        st.subheader("배치별 상세 (D+3~4)")

        if target.empty:
            st.info(f"{target_dates[0]} ~ {target_dates[1]} 수확 예정 배치가 없습니다.")
        else:
            disp = target.sort_values("harvest_date")[[
                "batch_id", "bed_type", "bed_id",
                "sow_date", "harvest_date", "total_days",
                "tray_or_gutter", "weight_per_plant_g", "loss_rate",
                "predicted_plants", "predicted_kg",
                "actual_yield", "actual_weight_kg", "actual_wpg",
                "note",
            ]].copy()

            disp["sow_date"]     = disp["sow_date"].apply(lambda v: fmt_d(v))
            disp["harvest_date"] = disp["harvest_date"].apply(lambda v: fmt_d(v))
            disp["total_days"]   = disp["total_days"].apply(
                lambda v: f"{int(v)}일" if pd.notna(v) else "—")
            disp["bed_type"]     = disp["bed_type"].map({"fixed": "고정", "mgs": "MGS"})
            disp["loss_rate"]    = disp["loss_rate"].apply(lambda v: f"{v*100:.0f}%")
            disp["weight_per_plant_g"] = disp["weight_per_plant_g"].apply(
                lambda v: f"{int(v)}g" if pd.notna(v) else "—")
            disp["predicted_plants"] = disp["predicted_plants"].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) else "N/A")
            disp["predicted_kg"] = disp["predicted_kg"].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "N/A")
            disp["actual_yield"] = disp["actual_yield"].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) else "—")
            disp["actual_weight_kg"] = disp["actual_weight_kg"].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            disp["actual_wpg"] = disp["actual_wpg"].apply(
                lambda v: f"{v:.1f}g" if pd.notna(v) else "—")

            disp.columns = [
                "배치 ID", "방식", "재배대/구역",
                "파종일", "수확예정일", "총 재배일",
                "판/거터", "주당 무게", "로스율",
                "예측 주수", "예측 무게(kg)",
                "실제 주수", "실제 무게(kg)", "실제 주당무게",
                "비고",
            ]
            st.dataframe(disp, use_container_width=True, hide_index=True)

            # 노션 마크다운
            with st.expander("📋 노션 마크다운 복사"):
                md_lines = [
                    f"## 수확량 예측 — {pred_date} 기준",
                    "",
                    "| 수확예정일 | 방식 | 재배대 | 파종일 | 수확예정일 | 총 재배일 | 예측 주수 | 예측 무게(kg) | 주당 무게 | 로스율 | 실제 무게(kg) | 실제 주당무게 | 비고 |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
                ]
                for _, row in target.sort_values("harvest_date").iterrows():
                    hd   = fmt_d(row["harvest_date"])
                    sd   = fmt_d(row["sow_date"])
                    td   = f"{int(row['total_days'])}일" if pd.notna(row["total_days"]) else "—"
                    bt   = "고정" if row["bed_type"] == "fixed" else "MGS"
                    pp   = f"{int(row['predicted_plants']):,}" if pd.notna(row["predicted_plants"]) else "N/A"
                    pk   = f"{row['predicted_kg']:.1f}" if pd.notna(row["predicted_kg"]) else "N/A"
                    wpg  = f"{int(row['weight_per_plant_g'])}g" if pd.notna(row["weight_per_plant_g"]) else "—"
                    loss = f"{row['loss_rate']*100:.0f}%"
                    ak   = f"{row['actual_weight_kg']:.1f}" if pd.notna(row["actual_weight_kg"]) else "—"
                    awpg = f"{row['actual_wpg']:.1f}g" if pd.notna(row["actual_wpg"]) else "—"
                    note = str(row.get("note","")) if pd.notna(row.get("note","")) else ""
                    md_lines.append(
                        f"| {hd} | {bt} | {row['bed_id']} | {sd} | {hd} | {td} | {pp}주 | {pk} | {wpg} | {loss} | {ak} | {awpg} | {note} |"
                    )
                md_lines.append(f"| **합계** | | | | | | **{total_pp:,}주** | **{total_pk:.1f}** | | | | | |")
                st.code("\n".join(md_lines), language="markdown")

# ──────────────────────────────────────────────────────────────
# TAB 2: 달별 전체 뷰
# ──────────────────────────────────────────────────────────────
with tab2:
    db_raw = get_db()
    if db_raw.empty:
        st.info("배치 DB가 비어 있습니다.")
    else:
        db = prepare_db(db_raw, loss_rate, default_weight_g, mgs_weight_g)
        db = add_predictions(db)
        df_m = db[db["harvest_date"].notna()].copy()
        df_m["ym"] = df_m["harvest_date"].dt.to_period("M")
        months = sorted(df_m["ym"].unique())

        if not months:
            st.info("수확 예정일이 있는 배치가 없습니다.")
        else:
            total_all_pp = int(df_m["predicted_plants"].sum(skipna=True))
            total_all_pk = round(float(df_m["predicted_kg"].sum(skipna=True)), 1)

            c1, c2, c3 = st.columns(3)
            c1.metric("전체 예측 주수", f"{total_all_pp:,}주")
            c2.metric("전체 예측 무게", f"{total_all_pk:.1f} kg")
            c3.metric("기간", f"{months[0]} ~ {months[-1]}")
            st.markdown("---")

            for ym in months:
                sub = df_m[df_m["ym"] == ym].sort_values("harvest_date")
                m_pp = int(sub["predicted_plants"].sum(skipna=True))
                m_pk = round(float(sub["predicted_kg"].sum(skipna=True)), 1)
                m_ap = int(sub["actual_yield"].sum(skipna=True)) if sub["actual_yield"].notna().any() else None
                m_ak = round(float(sub["actual_weight_kg"].sum(skipna=True)), 1) \
                       if sub["actual_weight_kg"].notna().any() else None

                with st.expander(
                    f"**{ym.year}년 {ym.month}월** — 예측 {m_pp:,}주 / {m_pk:.1f} kg  "
                    f"{'· 실적 ' + str(m_ak) + ' kg' if m_ak else ''}",
                    expanded=True,
                ):
                    mc = st.columns(5)
                    mc[0].metric("예측 주수", f"{m_pp:,}주")
                    mc[1].metric("예측 무게", f"{m_pk:.1f} kg")
                    mc[2].metric("실제 주수", f"{m_ap:,}주" if m_ap else "—")
                    mc[3].metric("실제 무게", f"{m_ak:.1f} kg" if m_ak else "—",
                                 delta=f"{round(m_ak-m_pk,1):+.1f} kg" if m_ak else None)
                    mc[4].metric("배치 수", f"{len(sub)}건")

                    disp = sub[[
                        "batch_id", "bed_type", "bed_id",
                        "sow_date", "harvest_date", "total_days",
                        "tray_or_gutter", "weight_per_plant_g", "loss_rate",
                        "predicted_plants", "predicted_kg",
                        "actual_yield", "actual_weight_kg", "actual_wpg",
                        "note",
                    ]].copy()
                    disp["sow_date"]     = disp["sow_date"].apply(fmt_d)
                    disp["harvest_date"] = disp["harvest_date"].apply(fmt_d)
                    disp["total_days"]   = disp["total_days"].apply(
                        lambda v: f"{int(v)}일" if pd.notna(v) else "—")
                    disp["bed_type"]     = disp["bed_type"].map({"fixed": "고정", "mgs": "MGS"})
                    disp["loss_rate"]    = disp["loss_rate"].apply(lambda v: f"{v*100:.0f}%")
                    disp["weight_per_plant_g"] = disp["weight_per_plant_g"].apply(
                        lambda v: f"{int(v)}g" if pd.notna(v) else "—")
                    disp["predicted_plants"] = disp["predicted_plants"].apply(
                        lambda v: f"{int(v):,}" if pd.notna(v) else "N/A")
                    disp["predicted_kg"]     = disp["predicted_kg"].apply(
                        lambda v: f"{v:.1f}" if pd.notna(v) else "N/A")
                    disp["actual_yield"]     = disp["actual_yield"].apply(
                        lambda v: f"{int(v):,}" if pd.notna(v) else "—")
                    disp["actual_weight_kg"] = disp["actual_weight_kg"].apply(
                        lambda v: f"{v:.1f}" if pd.notna(v) else "—")
                    disp["actual_wpg"]       = disp["actual_wpg"].apply(
                        lambda v: f"{v:.1f}g" if pd.notna(v) else "—")
                    disp.columns = [
                        "배치 ID", "방식", "재배대/구역",
                        "파종일", "수확예정일", "총 재배일",
                        "판/거터", "주당 무게", "로스율",
                        "예측 주수", "예측 무게(kg)",
                        "실제 주수", "실제 무게(kg)", "실제 주당무게",
                        "비고",
                    ]
                    st.dataframe(disp, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
# TAB 3: 배치 DB 관리
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("배치 DB 관리")
    db = get_db()

    mode = st.radio("작업 선택", ["조회", "배치 추가", "실적 수정", "배치 삭제"],
                    horizontal=True, key="tab3_mode")

    if mode == "조회":
        st.dataframe(db, use_container_width=True, hide_index=True)
        st.caption(f"총 {len(db)}건")

    elif mode == "배치 추가":
        st.markdown("**새 배치 입력**")
        with st.form("add_batch"):
            c1, c2, c3 = st.columns(3)
            batch_id  = c1.text_input("배치 ID *", placeholder="BATCH-F-01")
            bed_type  = c2.selectbox("재배 방식 *", ["fixed", "mgs"])
            bed_id    = c3.text_input("재배대/구역 *", placeholder="1 또는 MGS-A")

            c4, c5, c6 = st.columns(3)
            sow_date      = c4.date_input("파종일 *", value=date.today() - timedelta(25))
            transplant_dt = c5.date_input("이식일 (없으면 건너뜀)",
                                           value=date.today() - timedelta(18))
            plant_date    = c6.date_input("정식일 *", value=date.today() - timedelta(17))

            c7, c8 = st.columns(2)
            harvest_date = c7.date_input("수확 예정일 *", value=date.today() + timedelta(4))
            grow_days    = c8.number_input("생육일수 (정식→수확)", min_value=1, value=17)

            c9, c10, c11 = st.columns(3)
            tray_or_gutter    = c9.number_input("판 수 / 거터 수 *", min_value=1, value=40)
            weight_per_plant  = c10.number_input("주당 무게 (g)", min_value=1, value=100,
                                                  help="빈칸이면 기본값 적용")
            loss_rate_input   = c11.number_input("개별 로스율 (%)",
                                                  min_value=0, max_value=100, value=0,
                                                  help="0이면 전체 설정 로스율 적용")
            note_input = st.text_input("비고")

            submitted = st.form_submit_button("✅ 배치 추가")
            if submitted:
                if not batch_id or not bed_id:
                    st.error("배치 ID와 재배대/구역은 필수입니다.")
                else:
                    new_row = {
                        "batch_id":           batch_id,
                        "sow_date":           str(sow_date),
                        "transplant_date":    str(transplant_dt),
                        "plant_date":         str(plant_date),
                        "harvest_date":       str(harvest_date),
                        "grow_days":          grow_days,
                        "bed_type":           bed_type,
                        "bed_id":             bed_id,
                        "tray_or_gutter":     tray_or_gutter,
                        "weight_per_plant_g": weight_per_plant,
                        "loss_rate":          loss_rate_input / 100 if loss_rate_input > 0 else None,
                        "actual_yield":       None,
                        "actual_weight_kg":   None,
                        "note":               note_input,
                    }
                    db = db[db["batch_id"] != batch_id]  # 중복 덮어쓰기
                    db = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
                    set_db(db)
                    st.success(f"✅ {batch_id} 추가 완료 (DB 총 {len(db)}건)")

    elif mode == "실적 수정":
        st.markdown("**실적 직접 수정**")
        db_edit = get_db().copy()
        batch_ids = db_edit["batch_id"].tolist()
        if not batch_ids:
            st.info("배치가 없습니다.")
        else:
            sel = st.selectbox("배치 선택", batch_ids, key="tab3_sel_batch")
            row = db_edit[db_edit["batch_id"] == sel].iloc[0]
            with st.form("edit_actual"):
                ea, ek = st.columns(2)
                act_y = ea.number_input("실제 수확 주수",
                                         value=int(row["actual_yield"]) if pd.notna(row["actual_yield"]) else 0,
                                         min_value=0)
                act_k = ek.number_input("실제 수확 무게 (kg)",
                                         value=float(row["actual_weight_kg"]) if pd.notna(row["actual_weight_kg"]) else 0.0,
                                         min_value=0.0, step=0.1, format="%.1f")
                if st.form_submit_button("💾 저장"):
                    db_edit.loc[db_edit["batch_id"] == sel, "actual_yield"]     = act_y if act_y > 0 else None
                    db_edit.loc[db_edit["batch_id"] == sel, "actual_weight_kg"] = act_k if act_k > 0 else None
                    set_db(db_edit)
                    awpg_val = round(act_k * 1000 / act_y, 1) if act_y > 0 and act_k > 0 else None
                    st.success(f"✅ {sel} 실적 저장 완료" +
                               (f" · 실제 주당무게: {awpg_val}g" if awpg_val else ""))

    elif mode == "배치 삭제":
        db_now = get_db()
        if db_now.empty:
            st.info("삭제할 배치가 없습니다.")
        else:
            del_ids = st.multiselect("삭제할 배치 선택", db_now["batch_id"].tolist(), key="tab3_del_ids")
            if del_ids and st.button("🗑️ 삭제", type="primary", key="tab3_del_btn"):
                db_now = db_now[~db_now["batch_id"].isin(del_ids)]
                set_db(db_now)
                st.success(f"🗑️ {len(del_ids)}건 삭제 완료 (DB 총 {len(db_now)}건)")

# ──────────────────────────────────────────────────────────────
# TAB 4: 실적 입력 (빠른 일괄 입력)
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("실적 입력")
    st.caption("수확 완료 후 실제 주수와 무게를 입력하면 대시보드에 즉시 반영됩니다.")
    db_now = get_db()
    if db_now.empty:
        st.info("배치가 없습니다.")
    else:
        for col in ["actual_yield", "actual_weight_kg"]:
            db_now[col] = pd.to_numeric(db_now[col], errors="coerce")

        # 미입력 배치만 표시 (실적이 없는 것 우선)
        show_all = st.checkbox("실적 완료 배치도 표시", value=False, key="tab4_show_all")
        if show_all:
            candidates = db_now
        else:
            candidates = db_now[
                db_now["actual_weight_kg"].isna() | (db_now["actual_weight_kg"] == 0)
            ]

        if candidates.empty:
            st.success("모든 배치에 실적이 입력되어 있습니다.")
        else:
            # ── st.form 없이 위젯 직접 렌더링 ──────────────────
            # key에 탭 접두사 + 행 인덱스를 포함해 중복 방지
            input_vals = {}
            for idx, (_, row) in enumerate(candidates.iterrows()):
                bid = row["batch_id"]
                hd  = fmt_d(pd.to_datetime(row["harvest_date"], errors="coerce"))
                st.markdown(f"**{bid}** — 수확예정 {hd} | {row.get('bed_id', '?')}번")
                c1, c2 = st.columns(2)
                ay = c1.number_input(
                    f"실제 주수",
                    value=int(row["actual_yield"]) if pd.notna(row["actual_yield"]) else 0,
                    min_value=0,
                    key=f"t4_ay_{idx}_{bid}",
                )
                ak = c2.number_input(
                    f"실제 무게 (kg)",
                    value=float(row["actual_weight_kg"]) if pd.notna(row["actual_weight_kg"]) else 0.0,
                    min_value=0.0, step=0.1, format="%.1f",
                    key=f"t4_ak_{idx}_{bid}",
                )
                input_vals[bid] = (ay, ak)
                st.markdown("---")

            if st.button("💾 일괄 저장", key="tab4_save_btn", type="primary"):
                db_upd = get_db().copy()
                saved = 0
                for bid, (ay, ak) in input_vals.items():
                    if ay > 0 or ak > 0:
                        db_upd.loc[db_upd["batch_id"] == bid, "actual_yield"]     = ay if ay > 0 else None
                        db_upd.loc[db_upd["batch_id"] == bid, "actual_weight_kg"] = ak if ak > 0 else None
                        saved += 1
                set_db(db_upd)
                st.success(f"✅ {saved}건 저장 완료 → 대시보드 탭에서 확인하세요")

# ──────────────────────────────────────────────────────────────
# TAB 5: 참고 용량표
# ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader("고정 재배대 전체 용량표")
    yield_rate = 1 - loss_rate
    cap_rows = []
    for bid, trays in FIXED_BED_CONFIG.items():
        max_p  = trays * PLANTS_PER_TRAY
        pred_p = round(max_p * yield_rate)
        pred_k = round(pred_p * default_weight_g / 1000, 1)
        cap_rows.append({
            "재배대":               f"{bid}번",
            "판 수":                trays,
            "최대 식재 주수":        max_p,
            f"예측 주수 ({100-loss_pct}%)": pred_p,
            f"예측 kg ({default_weight_g}g/주)": pred_k,
        })
    cap_df = pd.DataFrame(cap_rows)
    tot = {
        "재배대":               "합계",
        "판 수":                cap_df["판 수"].sum(),
        "최대 식재 주수":        cap_df["최대 식재 주수"].sum(),
        f"예측 주수 ({100-loss_pct}%)": cap_df[f"예측 주수 ({100-loss_pct}%)"].sum(),
        f"예측 kg ({default_weight_g}g/주)": round(cap_df[f"예측 kg ({default_weight_g}g/주)"].sum(), 1),
    }
    cap_df = pd.concat([cap_df, pd.DataFrame([tot])], ignore_index=True)
    st.dataframe(cap_df, use_container_width=True, hide_index=True)
    st.caption(f"1판당 {PLANTS_PER_TRAY}주 · MGS 거터당 {PLANTS_PER_GUTTER}주")
