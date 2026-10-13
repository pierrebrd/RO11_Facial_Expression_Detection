from typing import Optional, List, Dict, Any
import pandas as pd
import altair as alt
import streamlit as st

from models import CANONICAL_EMOTIONS, EMOTION_EMOJIS
from pipeline import ModelResult, FrameAnalysisResult, LatencyTracker


def render_model_card(
    res: Optional[ModelResult],
    col_title: str,
    tracker_stats: Optional[Dict[str, Optional[float]]] = None
):
    """
    Renders prediction statistics, latencies (instant and mean), FPS, and spaced emotion bars.
    """
    st.subheader(col_title)

    if res is None:
        st.info("No face detected or model disabled.")
        return

    # Emotion prediction with emotion emoji
    st.markdown(f"### {res.emoji} {res.top_emotion} ({res.top_prob * 100:.1f}%)")

    # Latency and FPS metrics (Instant + Mean)
    fps = 1000.0 / res.total_latency_ms if res.total_latency_ms > 0 else 0.0

    delta_img = f"Mean: {tracker_stats['img_proc']:.1f} ms" if tracker_stats and tracker_stats.get("img_proc") is not None else None
    delta_net = f"Mean: {tracker_stats['network']:.1f} ms" if tracker_stats and tracker_stats.get("network") is not None else None
    delta_tot = f"Mean: {tracker_stats['total']:.1f} ms" if tracker_stats and tracker_stats.get("total") is not None else None
    delta_fps = f"Mean: {tracker_stats['fps']:.1f}" if tracker_stats and tracker_stats.get("fps") is not None else None

    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Image Proc Latency", value=f"{res.image_proc_latency_ms:.1f} ms", delta=delta_img, delta_color="off")
        st.metric(label="Total Latency", value=f"{res.total_latency_ms:.1f} ms", delta=delta_tot, delta_color="off")
    with m2:
        st.metric(label="Network Latency", value=f"{res.network_latency_ms:.1f} ms", delta=delta_net, delta_color="off")
        st.metric(label="Throughput", value=f"{fps:.1f} FPS", delta=delta_fps, delta_color="off")

    st.caption(
        f"Haar + 48px: {res.haar_latency_ms:.1f} ms | "
        f"Feature Prep: {res.feature_prep_latency_ms:.1f} ms"
    )

    # Emotion Confidence Bar Chart with generous height and spacing
    df_probs = pd.DataFrame([
        {
            "Emotion": f"{EMOTION_EMOJIS.get(emo, '')} {emo}",
            "Confidence": round(res.probabilities.get(emo, 0.0) * 100.0, 1)
        }
        for emo in CANONICAL_EMOTIONS
    ])

    bars = (
        alt.Chart(df_probs)
        .mark_bar(size=14, cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Confidence:Q",
                scale=alt.Scale(domain=[0, 110]),
                title="Confidence (%)"
            ),
            y=alt.Y(
                "Emotion:N",
                sort=None,
                title=None,
                axis=alt.Axis(
                    labelOverlap=False,
                    labelLimit=300,
                    labelFontSize=12,
                    labelPadding=6
                )
            ),
            color=alt.Color(
                "Confidence:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None
            ),
            tooltip=["Emotion", alt.Tooltip("Confidence:Q", format=".1f")]
        )
    )

    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        fontSize=11
    ).encode(
        text=alt.Text("Confidence:Q", format=".1f")
    )

    chart = (bars + text).properties(height=280)
    st.altair_chart(chart, width='stretch',)


def render_active_models(
    analysis: FrameAnalysisResult,
    enabled_models: List[str],
    tracker: Optional[LatencyTracker] = None
):
    """
    Renders model cards dynamically for the enabled models.
    """
    if not enabled_models:
        st.warning("No models enabled. Check at least one model above.")
        return

    model_registry = {
        "cnn": ("Mini-Xception (CNN)", analysis.cnn),
        "svm": ("HOG + Linear SVM", analysis.svm),
        "knn": ("LBP + KNN", analysis.knn),
    }

    active_items = [
        (key, model_registry[key][0], model_registry[key][1])
        for key in ["cnn", "svm", "knn"]
        if key in enabled_models
    ]

    cols = st.columns(len(active_items))
    for col, (key, title, res) in zip(cols, active_items):
        stats = tracker.get_means(key) if tracker else None
        with col:
            render_model_card(res, title, tracker_stats=stats)


def render_latency_summary_table(tracker: Optional[LatencyTracker], enabled_models: List[str]):
    """
    Renders the table of mean latencies below the current results cards.
    """
    if tracker is None or not enabled_models:
        return

    df = tracker.get_summary_dataframe(enabled_models)
    if not df.empty:
        st.markdown("#### Mean Latency Benchmark (Running Average)")
        st.dataframe(df, width='stretch', hide_index=True)
