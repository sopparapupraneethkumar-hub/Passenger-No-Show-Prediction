import inspect
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_DIR = "C:/Users/prane/Downloads/project16_outputs"
RED, BLUE, GREY = "#d62728", "#1f77b4", "#7f7f7f"
WINDOW_ORDER = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FILES = ["predictions.csv", "metrics.csv", "model_comparison.csv", "confusion_matrix.csv",
         "feature_coefficients.csv", "eda_segments.csv", "eda_monthly.csv"]
DIM_LABELS = {"hotel": "Hotel", "customer_type": "Customer type", "ticket_type": "Ticket type (deposit type)",
              "booking_window": "Booking window", "market_segment": "Market segment",
              "distribution_channel": "Distribution channel"}

st.set_page_config(page_title="Passenger No-Show Prediction", layout="wide")

# Streamlit changed the way chart/table width is set; support both old and new versions.
_ALT_KW = {"width": "stretch"} if "width" in inspect.signature(st.altair_chart).parameters else {"use_container_width": True}
_DF_KW = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}


def show(chart):
    st.altair_chart(chart, **_ALT_KW)


def table(df):
    st.dataframe(df, hide_index=True, **_DF_KW)


@st.cache_data
def load_all(folder, stamp):
    base = Path(folder)
    return {f[:-4]: pd.read_csv(base / f) for f in FILES}


def confusion_counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def safe_div(a, b):
    return a / b if b else 0.0


def roc_points(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-p, kind="mergesort")
    y_s, p_s = y[order], p[order]
    tps, fps = np.cumsum(y_s), np.cumsum(1 - y_s)
    last = np.r_[np.where(np.diff(p_s))[0], len(p_s) - 1]
    tpr = np.r_[0.0, tps[last] / tps[-1]]
    fpr = np.r_[0.0, fps[last] / fps[-1]]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return pd.DataFrame({"fpr": fpr, "tpr": tpr}), auc


def grouped_bars(df, cat, order=None, title=""):
    long = df.melt(id_vars=[cat], value_vars=["Actual %", "Predicted %"], var_name="Series", value_name="No-show %")
    return (alt.Chart(long, title=title).mark_bar()
            .encode(x=alt.X(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset="Series:N",
                    y=alt.Y("No-show %:Q", title="No-show %"),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual %", "Predicted %"], range=[RED, BLUE])),
                    tooltip=[cat, "Series", alt.Tooltip("No-show %:Q", format=".1f")])
            .properties(height=320))


# ------------------------------------------------------------------ load data
st.sidebar.title("Passenger No-Show")
folder = st.sidebar.text_input("Output folder", os.environ.get("PROJECT16_OUT", DEFAULT_DIR))
missing = [f for f in FILES if not (Path(folder) / f).exists()]
if missing:
    st.title("Passenger No-Show Prediction")
    st.error("These files were not found in: " + folder)
    st.code("\n".join(missing))
    st.info("Run the PySpark code first (it creates these files), or fix the folder path in the sidebar.")
    st.stop()

stamp = tuple((Path(folder) / f).stat().st_mtime for f in FILES)
D = load_all(folder, stamp)
M = dict(zip(D["metrics"]["metric"], D["metrics"]["value"]))
pred_all = D["predictions"]

# ------------------------------------------------------------------ sidebar controls
model_choice = st.sidebar.radio("Model used in the charts", ["Standard", "Class-weighted"],
                                help="Class-weighted catches more no-shows (higher recall) but raises more false alarms.")
prob_col = "noshow_prob" if model_choice == "Standard" else "noshow_prob_weighted"
st.sidebar.markdown("**Filter the test bookings**")
mask = pd.Series(True, index=pred_all.index)
for col, label in [("hotel", "Hotel"), ("customer_type", "Customer type"),
                   ("ticket_type", "Ticket type"), ("booking_window", "Booking window")]:
    options = [w for w in WINDOW_ORDER if w in set(pred_all[col])] if col == "booking_window" else sorted(pred_all[col].unique())
    chosen = st.sidebar.multiselect(label, options, default=options)
    mask &= pred_all[col].isin(chosen)
pred = pred_all[mask].copy()
st.sidebar.caption(f"{len(pred):,} of {len(pred_all):,} test bookings selected")

st.title("Passenger No-Show Prediction Dashboard")
st.caption("PySpark MLlib Logistic Regression | Hotel Booking Demand data used as a stand-in for passenger bookings | "
           "'No-show' = booking cancelled or guest did not turn up")

if pred.empty:
    st.warning("No bookings match the current filters. Select at least one value in every filter.")
    st.stop()

pred["Risk band"] = pd.cut(pred[prob_col], [-0.01, 0.4, 0.7, 1.01], labels=["Low", "Medium", "High"])

tabs = st.tabs(["Overview", "Segment analysis", "Predicted vs actual", "Model performance",
                "What drives no-shows", "Trends", "Explore bookings"])

# ------------------------------------------------------------------ 1. overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Bookings (cleaned)", f"{int(M['clean_rows']):,}")
    c[1].metric("No-show rate", f"{M['noshow_rate_pct']:.1f}%")
    c[2].metric("Accuracy", f"{M['accuracy']:.1%}")
    c[3].metric("ROC-AUC", f"{M['roc_auc']:.3f}")
    c[4].metric("Precision (no-show)", f"{M['precision']:.1%}")
    c[5].metric("Recall (no-show)", f"{M['recall']:.1%}")
    st.caption("Model metrics are from Spark on the full test set, using the standard Logistic Regression at threshold 0.5.")
    left, right = st.columns(2)
    with left:
        st.subheader("Data cleaning")
        dup = int(M["duplicate_rows"])
        after_dup = int(M["raw_rows"]) - dup
        funnel = pd.DataFrame({"Stage": ["1. Raw rows", "2. After removing duplicates", "3. After removing invalid records"],
                               "Rows": [int(M["raw_rows"]), after_dup, int(M["clean_rows"])]})
        show(alt.Chart(funnel).mark_bar(color=BLUE).encode(
            y=alt.Y("Stage:N", sort=None, title=None), x=alt.X("Rows:Q"), tooltip=["Stage", "Rows"]).properties(height=170))
        st.write(f"Duplicates removed: **{dup:,}** | zero-night stays found: **{int(M['zero_night_rows']):,}** | "
                 f"bookings with no guests: **{int(M['no_guest_rows']):,}**")
    with right:
        st.subheader("Why accuracy alone is not enough")
        cmp_df = pd.DataFrame({"Model": ["Always predict 'shows up'", "Logistic Regression"],
                               "Accuracy": [M["baseline_accuracy"], M["accuracy"]]})
        show(alt.Chart(cmp_df).mark_bar().encode(
            x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=None, scale=alt.Scale(range=[GREY, BLUE])),
            tooltip=["Model", alt.Tooltip("Accuracy:Q", format=".1%")]).properties(height=170))
        st.write("A model that never predicts a no-show already scores "
                 f"**{M['baseline_accuracy']:.1%}** accuracy, so ROC-AUC and recall for the no-show class matter more.")

# ------------------------------------------------------------------ 2. segment analysis (full cleaned data)
with tabs[1]:
    st.subheader("No-show rate by segment (all cleaned bookings)")
    dim = st.selectbox("Group by", list(DIM_LABELS), format_func=lambda k: DIM_LABELS[k])
    seg = D["eda_segments"]
    seg = seg[seg["dimension"] == dim].copy()
    order = WINDOW_ORDER if dim == "booking_window" else seg.sort_values("noshow_pct", ascending=False)["segment"].tolist()
    bars = alt.Chart(seg).mark_bar(color=RED).encode(
        x=alt.X("segment:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("noshow_pct:Q", title="No-show %"),
        tooltip=["segment", alt.Tooltip("noshow_pct:Q", format=".1f", title="No-show %"), alt.Tooltip("bookings:Q", format=",")])
    avg_line = alt.Chart(pd.DataFrame({"y": [M["noshow_rate_pct"]]})).mark_rule(color=GREY, strokeDash=[6, 4]).encode(y="y:Q")
    show((bars + avg_line).properties(height=340))
    st.caption(f"Dashed line = overall no-show rate ({M['noshow_rate_pct']:.1f}%).")
    table(seg.sort_values("noshow_pct", ascending=False).rename(
        columns={"segment": "Segment", "bookings": "Bookings", "noshow_pct": "No-show %"})[["Segment", "Bookings", "No-show %"]])

# ------------------------------------------------------------------ 3. predicted vs actual
with tabs[2]:
    st.subheader(f"Predicted vs actual no-show % ({model_choice.lower()} model, filtered test bookings)")
    actual_pct = pred["actual_noshow"].mean() * 100
    predicted_pct = pred[prob_col].mean() * 100
    c = st.columns(3)
    c[0].metric("Actual no-show %", f"{actual_pct:.2f}%")
    c[1].metric("Predicted no-show % (average probability)", f"{predicted_pct:.2f}%",
                delta=f"{predicted_pct - actual_pct:+.2f} pts vs actual", delta_color="off")
    c[2].metric("Bookings", f"{len(pred):,}")
    dim = st.selectbox("Compare by", list(DIM_LABELS), format_func=lambda k: DIM_LABELS[k], key="pva_dim")
    g = pred.groupby(dim).agg(Bookings=("actual_noshow", "size"), **{"Actual %": ("actual_noshow", "mean"),
                                                                    "Predicted %": (prob_col, "mean")}).reset_index()
    g["Actual %"] *= 100
    g["Predicted %"] *= 100
    g["Gap (pts)"] = g["Predicted %"] - g["Actual %"]
    order = WINDOW_ORDER if dim == "booking_window" else None
    show(grouped_bars(g, dim, order, title=f"Actual vs predicted no-show % by {DIM_LABELS[dim].lower()}"))
    table(g.round(2))
    if model_choice == "Class-weighted":
        st.info("The class-weighted model deliberately over-predicts no-shows to catch more of them, "
                "so its average probability is higher than the real rate. Use the Standard model for calibrated percentages.")

# ------------------------------------------------------------------ 4. model performance
with tabs[3]:
    st.subheader("Model performance on the filtered test bookings")
    thr = st.slider("Decision threshold (predict 'no-show' when probability is above this)", 0.05, 0.95, 0.50, 0.01)
    y = pred["actual_noshow"].to_numpy().astype(int)
    p = pred[prob_col].to_numpy().astype(float)
    tp, fp, fn, tn = confusion_counts(y, (p > thr).astype(int))
    acc = safe_div(tp + tn, len(y))
    prec, rec = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1v = safe_div(2 * prec * rec, prec + rec)
    roc = roc_points(y, p)
    c = st.columns(5)
    c[0].metric("Accuracy", f"{acc:.1%}")
    c[1].metric("Precision", f"{prec:.1%}")
    c[2].metric("Recall", f"{rec:.1%}")
    c[3].metric("F1", f"{f1v:.3f}")
    c[4].metric("ROC-AUC", f"{roc[1]:.3f}" if roc else "n/a")
    left, right = st.columns(2)
    with left:
        cm = pd.DataFrame({"Actual": ["No-show", "No-show", "Shows up", "Shows up"],
                           "Predicted": ["No-show", "Shows up", "No-show", "Shows up"], "Count": [tp, fn, fp, tn]})
        base = alt.Chart(cm, title="Confusion matrix").encode(
            x=alt.X("Predicted:N", sort=["No-show", "Shows up"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", sort=["No-show", "Shows up"]))
        heat = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
        txt = base.mark_text(fontSize=20, fontWeight="bold").encode(text=alt.Text("Count:Q", format=","),
                                                                    color=alt.value("black"))
        show((heat + txt).properties(height=300))
    with right:
        if roc:
            curve, auc_val = roc
            line = alt.Chart(curve, title=f"ROC curve (AUC = {auc_val:.3f})").mark_line(color=BLUE, strokeWidth=3).encode(
                x=alt.X("fpr:Q", title="False positive rate"), y=alt.Y("tpr:Q", title="True positive rate (recall)"))
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=GREY, strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
            pt = alt.Chart(pd.DataFrame({"x": [safe_div(fp, fp + tn)], "y": [rec]})).mark_point(color=RED, size=140, filled=True).encode(x="x:Q", y="y:Q")
            show((line + diag + pt).properties(height=300))
            st.caption("Red dot = current threshold.")
        else:
            st.warning("ROC needs both outcomes in the selection.")
    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in ts:
        a, b, d_, e = confusion_counts(y, (p > t).astype(int))
        rows.append((t, safe_div(a, a + b), safe_div(a, a + d_)))
    pr = pd.DataFrame(rows, columns=["Threshold", "Precision", "Recall"]).melt("Threshold", var_name="Metric", value_name="Value")
    show(alt.Chart(pr, title="Precision and recall vs threshold").mark_line(strokeWidth=3).encode(
        x="Threshold:Q", y=alt.Y("Value:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=["Precision", "Recall"], range=[BLUE, RED])),
        tooltip=["Metric", alt.Tooltip("Threshold:Q"), alt.Tooltip("Value:Q", format=".1%")]).properties(height=280))
    st.markdown("**Standard vs class-weighted model (full test set, threshold 0.5, from Spark)**")
    table(D["model_comparison"].round(4))

# ------------------------------------------------------------------ 5. drivers
with tabs[4]:
    st.subheader("What pushes no-show risk up or down? (Logistic Regression)")
    n_top = st.slider("Number of features", 5, 30, 12)
    co = D["feature_coefficients"].copy()
    co["abs"] = co["coefficient"].abs()
    co = co.sort_values("abs", ascending=False).head(n_top)
    co["Effect"] = np.where(co["coefficient"] > 0, "Raises risk", "Lowers risk")
    show(alt.Chart(co).mark_bar().encode(
        y=alt.Y("feature:N", sort=alt.EncodingSortField("abs", order="descending"), title=None),
        x=alt.X("coefficient:Q", title="Coefficient (log-odds)"),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, BLUE])),
        tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f"), alt.Tooltip("odds_ratio:Q", format=".2f")]
    ).properties(height=max(260, 26 * n_top)))
    table(co[["feature", "coefficient", "odds_ratio", "Effect"]].round(3))
    st.caption("Odds ratio above 1 raises no-show risk, below 1 lowers it. Numeric features are standardized (effect per 1 standard "
               "deviation); category features show the effect of that category versus the others. These are associations in the data, not proven causes.")

# ------------------------------------------------------------------ 6. trends
with tabs[5]:
    st.subheader("Bookings and no-show rate by arrival month")
    mo = D["eda_monthly"].copy()
    mo["Month"] = pd.to_datetime(dict(year=mo["arrival_date_year"], month=mo["month_num"], day=1))
    show(alt.Chart(mo, title="Bookings per arrival month").mark_bar(color=BLUE).encode(
        x=alt.X("Month:T", title=None), y=alt.Y("bookings:Q", title="Bookings"),
        tooltip=[alt.Tooltip("Month:T", format="%b %Y"), "bookings"]).properties(height=260))
    show(alt.Chart(mo, title="No-show % per arrival month").mark_line(color=RED, point=alt.OverlayMarkDef(color=RED), strokeWidth=3).encode(
        x=alt.X("Month:T", title=None), y=alt.Y("noshow_pct:Q", title="No-show %"),
        tooltip=[alt.Tooltip("Month:T", format="%b %Y"), alt.Tooltip("noshow_pct:Q", format=".1f")]).properties(height=260))

# ------------------------------------------------------------------ 7. explore
with tabs[6]:
    st.subheader("Risk bands and highest-risk bookings")
    st.caption("Bands from the predicted probability of the selected model: Low below 0.4, Medium 0.4 to 0.7, High above 0.7.")
    bands = pred.groupby("Risk band", observed=True).agg(Bookings=("actual_noshow", "size"),
                                                         **{"Actual no-show %": ("actual_noshow", "mean")}).reset_index()
    bands["Actual no-show %"] *= 100
    left, right = st.columns([1, 2])
    with left:
        table(bands.round(1))
    with right:
        show(alt.Chart(bands).mark_bar().encode(
            x=alt.X("Risk band:N", sort=["Low", "Medium", "High"], title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual no-show %:Q"),
            color=alt.Color("Risk band:N", legend=None, scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#2ca02c", "#ff7f0e", RED])),
            tooltip=["Risk band", "Bookings", alt.Tooltip("Actual no-show %:Q", format=".1f")]).properties(height=220))
    st.write("A good model shows a much higher real no-show rate in the High band than in the Low band.")
    n_rows = st.slider("Show top N riskiest bookings", 10, 200, 25, 5)
    cols = ["hotel", "customer_type", "ticket_type", "booking_window", "market_segment", "lead_time", "total_nights", "adr",
            prob_col, "actual_noshow"]
    top = pred.sort_values(prob_col, ascending=False).head(n_rows)[cols].rename(
        columns={prob_col: "no-show probability", "actual_noshow": "actually no-show (1=yes)"})
    table(top.round(3))
    st.download_button("Download filtered predictions (CSV)", pred.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_predictions.csv", mime="text/csv")
