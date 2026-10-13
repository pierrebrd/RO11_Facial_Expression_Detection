from typing import Optional, List
import pandas as pd
import altair as alt
import streamlit as st

from models import CANONICAL_EMOTIONS, EMOTION_EMOJIS
from pipeline import ModelResult, FrameAnalysisResult


def render_model_card(res: Optional[ModelResult], col_title: str):
    """
    Renders prediction statistics, latencies, FPS, and spaced emotion bars.
    """
    st.subheader(col_title)

    if res is None:
        st.info("No face detected or model disabled.")
        return

    # Emotion prediction with emotion emoji
    st.markdown(f"### {res.emoji} {res.top_emotion} ({res.top_prob * 100:.1f}%)")

    # Latency and FPS metrics
    fps = 1000.0 / res.total_latency_ms if res.total_latency_ms > 0 else 0.0

    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Image Proc Latency", value=f"{res.image_proc_latency_ms:.1f} ms")
        st.metric(label="Total Latency", value=f"{res.total_latency_ms:.1f} ms")
    with m2:
        st.metric(label="Network Latency", value=f"{res.network_latency_ms:.1f} ms")
        st.metric(label="Throughput", value=f"{fps:.1f} FPS")

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
    st.altair_chart(chart, use_container_width=True)


def render_active_models(analysis: FrameAnalysisResult, enabled_models: List[str]):
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
        (model_registry[key][0], model_registry[key][1])
        for key in ["cnn", "svm", "knn"]
        if key in enabled_models
    ]

    cols = st.columns(len(active_items))
    for col, (title, res) in zip(cols, active_items):
        with col:
            render_model_card(res, title)
