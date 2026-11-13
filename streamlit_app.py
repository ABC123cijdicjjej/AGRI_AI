"""Public web interface for the saved PlantVillage CNN classifier."""

from hashlib import sha256
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from leaf_model import LABELS, display_label, load_model, predict_image


st.set_page_config(
    page_title="Verdant · Plant health",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_classifier():
    """Keep the loaded CNN in memory across Streamlit reruns."""
    return load_model()


st.html("""
<style>
    .verdant-hero {
        --hero-ink: #f5f7ef;
        --hero-muted: #c5d5c7;
        position: relative;
        overflow: hidden;
        isolation: isolate;
        padding: clamp(1.7rem, 4vw, 3.5rem);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 1.5rem;
        color: var(--hero-ink);
        background:
            radial-gradient(ellipse at 87% 14%, rgba(196, 218, 150, 0.2), transparent 28rem),
            radial-gradient(ellipse at 4% 100%, rgba(56, 121, 82, 0.55), transparent 32rem),
            linear-gradient(118deg, #123b2b 0%, #1b4c36 54%, #24553c 100%);
        box-shadow: 0 1.5rem 4rem rgba(26, 57, 39, 0.14);
        animation: verdant-arrive 650ms cubic-bezier(0.2, 0.7, 0.2, 1) both;
    }

    .verdant-hero::after {
        position: absolute;
        z-index: -1;
        top: -9rem;
        right: -5rem;
        width: 25rem;
        height: 25rem;
        border: 1px solid rgba(230, 239, 218, 0.12);
        border-radius: 50%;
        box-shadow:
            0 0 0 2.5rem rgba(230, 239, 218, 0.035),
            0 0 0 5rem rgba(230, 239, 218, 0.025);
        content: "";
        pointer-events: none;
    }

    .verdant-hero__layout {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(13rem, 0.65fr);
        align-items: end;
        gap: clamp(2rem, 6vw, 5rem);
        max-width: 76rem;
        margin: 0 auto;
    }

    .verdant-hero__eyebrow {
        margin: 0 0 1.25rem;
        color: #d7e5c7;
        font: 600 0.7rem/1.2 system-ui, sans-serif;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .verdant-hero h1 {
        max-width: 12ch;
        margin: 0;
        color: var(--hero-ink);
        font: 500 clamp(2.6rem, 5.2vw, 4.65rem)/0.99 Georgia, "Times New Roman", serif;
        letter-spacing: -0.045em;
        text-wrap: balance;
    }

    .verdant-hero__description {
        max-width: 39rem;
        margin: 1.25rem 0 0;
        color: var(--hero-muted);
        font: 400 clamp(0.95rem, 1.3vw, 1.08rem)/1.7 system-ui, sans-serif;
    }

    .verdant-hero__coverage {
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(238, 244, 230, 0.17);
        border-radius: 1rem;
        background: rgba(245, 249, 239, 0.075);
        backdrop-filter: blur(12px);
    }

    .verdant-hero__coverage-label {
        margin: 0 0 1rem;
        color: #d7e5c7;
        font: 600 0.66rem/1.2 system-ui, sans-serif;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .verdant-hero__fact {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.72rem 0;
        border-top: 1px solid rgba(238, 244, 230, 0.13);
        color: var(--hero-muted);
        font: 400 0.82rem/1.3 system-ui, sans-serif;
    }

    .verdant-hero__fact strong {
        color: var(--hero-ink);
        font: 500 1.08rem/1 Georgia, "Times New Roman", serif;
        white-space: nowrap;
    }

    @keyframes verdant-arrive {
        from { opacity: 0; transform: translateY(0.7rem); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 680px) {
        .verdant-hero {
            border-radius: 1.1rem;
        }

        .verdant-hero__layout {
            grid-template-columns: 1fr;
            align-items: start;
            gap: 1.5rem;
        }

        .verdant-hero__coverage {
            max-width: 24rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .verdant-hero {
            animation: none;
        }
    }
</style>
<section class="verdant-hero" aria-labelledby="verdant-title">
    <div class="verdant-hero__layout">
        <div>
            <p class="verdant-hero__eyebrow">Verdant · Plant health studio</p>
            <h1 id="verdant-title">A clearer picture of plant health.</h1>
            <p class="verdant-hero__description">
                A thoughtful first look at leaf health, powered by a focused
                visual model. Start with one clear photo; leave with a useful
                shortlist.
            </p>
        </div>
        <aside class="verdant-hero__coverage" aria-label="Model coverage">
            <p class="verdant-hero__coverage-label">A focused screening model</p>
            <div class="verdant-hero__fact"><span>Plant types</span><strong>05</strong></div>
            <div class="verdant-hero__fact"><span>Leaf conditions</span><strong>25</strong></div>
            <div class="verdant-hero__fact"><span>Image input</span><strong>128 px</strong></div>
        </aside>
    </div>
</section>
""")


def render_result(ranked):
    predicted_label, confidence = ranked[0]
    crop, condition = predicted_label.split("___", 1)
    healthy = condition.strip().lower() == "healthy"

    st.badge(
        "Healthy signal" if healthy else "Needs a closer look",
        icon=":material/check_circle:" if healthy else ":material/health_and_safety:",
        color="green" if healthy else "orange",
    )
    st.space("small")
    st.caption("LEADING MATCH · {}".format(display_label(crop).upper()))
    st.subheader(display_label(condition), anchor=False)
    st.metric("Model score", "{:.1f}%".format(confidence * 100))
    st.progress(
        max(0.0, min(confidence, 1.0)),
        text="Relative model score",
    )

    st.space("small")
    st.markdown("**Other possible matches**")
    for position, (label, score) in enumerate(ranked[1:], start=2):
        condition_name = display_label(label.split("___", 1)[1])
        with st.container(border=True):
            st.markdown(
                "**{:02d}**　{}　·　{:.1f}%".format(
                    position, condition_name, score * 100
                )
            )
            st.progress(max(0.0, min(score, 1.0)), text="Relative model score")
    st.caption(
        "Scores are model estimates, not calibrated probabilities. "
        "Confirm suspected disease with a local expert."
    )


st.space("medium")
st.subheader("Start with a leaf", anchor=False)
st.caption(
    "Upload one clear photo to screen for apple, corn, grape, potato, or tomato conditions."
)
st.space("small")

upload_column, result_column = st.columns([1.05, 0.95], gap="large")
uploaded_file = None
leaf_image = None
upload_digest = None
analyze_clicked = False

with upload_column:
    with st.container(border=True):
        st.badge("01  /  ADD A PHOTO", color="green")
        st.subheader("Your leaf sample", anchor=False)
        st.caption("A single, well-lit leaf gives the model the best view.")
        uploaded_file = st.file_uploader(
            "Upload a leaf photo",
            type=("jpg", "jpeg", "png"),
            max_upload_size=10,
            help="JPG or PNG, up to 10 MB.",
            key="leaf_upload",
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            upload_digest = sha256(image_bytes).hexdigest()
            try:
                with Image.open(BytesIO(image_bytes)) as source:
                    source.verify()
                with Image.open(BytesIO(image_bytes)) as source:
                    leaf_image = source.convert("RGB")
                st.image(
                    leaf_image,
                    alt="Preview of the uploaded plant leaf",
                    width="stretch",
                )
                st.caption(
                    "{}  ·  {} × {} px".format(
                        uploaded_file.name, leaf_image.width, leaf_image.height
                    )
                )
            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(
                    "This file could not be opened as an image. "
                    "Choose a valid JPG or PNG."
                )
                st.caption(str(error))

        if upload_digest != st.session_state.get("upload_digest"):
            st.session_state.pop("analysis", None)
            st.session_state["upload_digest"] = upload_digest

        analyze_clicked = st.button(
            "Screen this leaf",
            type="primary",
            icon=":material/arrow_right_alt:",
            width="stretch",
            disabled=leaf_image is None,
            key="analyze_leaf",
        )
        st.caption(
            "Private by design · Photos are analyzed in memory and aren't saved."
        )

with result_column:
    with st.container(border=True):
        st.badge("02  /  YOUR SCREENING", color="green")
        st.subheader("The first read", anchor=False)
        st.caption("A leading match and two alternatives from the 25-class model.")

        analysis = st.session_state.get("analysis")
        if analyze_clicked and leaf_image is not None:
            try:
                with st.spinner("Preparing your screening…"):
                    ranked = predict_image(leaf_image, model=get_classifier())
                analysis = {"digest": upload_digest, "ranked": ranked}
                st.session_state["analysis"] = analysis
            except Exception as error:
                st.error(
                    "Your screening couldn't be completed. Check that the model "
                    "files and runtime dependencies are available, then try again."
                )
                st.caption(str(error))
                analysis = None

        if analysis and analysis.get("digest") == upload_digest:
            render_result(analysis["ranked"])
        elif leaf_image is not None:
            st.info(
                "Your image is ready. Select **Screen this leaf** to see "
                "the leading match.",
                title="Ready for a closer look",
                icon=":material/energy_savings_leaf:",
            )
        else:
            st.info(
                "Add a clear photo to begin. Your screening will appear here.",
                title="Your results will appear here",
                icon=":material/energy_savings_leaf:",
            )

st.space("large")
st.subheader("Designed for a careful first look", anchor=False)
st.caption(
    "A simple workflow that keeps the model's role—and its limits—clear."
)
step_columns = st.columns(3, gap="medium")
steps = (
    ("01", "Share a clear view", "Choose a bright photo with one leaf in focus."),
    ("02", "Screen the pattern", "The CNN compares it with 25 visual classes."),
    ("03", "Use your judgment", "Treat the result as a prompt, not a diagnosis."),
)
for column, (number, title, description) in zip(step_columns, steps):
    with column:
        with st.container(border=True, height="stretch"):
            st.badge(number, color="green")
            st.markdown("**{}**".format(title))
            st.caption(description)

st.space("medium")
with st.container(border=True):
    st.caption(
        "**A screening aid, not a diagnosis.** Predictions may be incorrect "
        "or incomplete. For treatment decisions, consult a local agricultural "
        "extension service or plant-health professional."
    )
