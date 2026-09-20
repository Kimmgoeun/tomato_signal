
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Tomato Signal",
    page_icon="🍅"
)

st.title("🍅 Tomato Signal")

st.write(
    "토마토 잎 사진을 업로드하면 "
    "정상 / 초기역병 / 잎곰팡이병 / "
    "황화잎말림바이러스를 분류합니다."
)

MODEL_PATH = "tomato_4class_clean_v3_stage2_best.keras"

CLASS_NAMES = [
    "정상",
    "초기역병",
    "잎곰팡이병",
    "황화잎말림바이러스"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()


uploaded = st.file_uploader(
    "토마토 잎 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"]
)


if uploaded is not None:

    image = Image.open(
        uploaded
    ).convert("RGB")

    st.image(
        image,
        caption="업로드 이미지",
        use_container_width=True
    )

    resized = image.resize(
        (224, 224)
    )

    x = np.array(
        resized,
        dtype=np.float32
    )

    x = np.expand_dims(
        x,
        axis=0
    )

    probs = model.predict(
        x,
        verbose=0
    )[0]

    pred_idx = int(
        np.argmax(probs)
    )

    max_prob = float(
        probs[pred_idx]
    )


    # -------------------------------
    # 신뢰도 낮으면 판단 보류
    # -------------------------------

    if max_prob < 0.60:

        st.warning(
            "⚠️ 지원 범위 외 증상이거나 "
            "판별 신뢰도가 낮습니다."
        )

        st.write(
            f"가장 가까운 예측: "
            f"{CLASS_NAMES[pred_idx]} "
            f"({max_prob * 100:.1f}%)"
        )

    else:

        st.success(
            f"예측 결과: "
            f"{CLASS_NAMES[pred_idx]}"
        )

        st.write(
            f"신뢰도: "
            f"{max_prob * 100:.1f}%"
        )


    st.subheader(
        "클래스별 예측 확률"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        st.write(
            f"{name}: "
            f"{probs[i] * 100:.1f}%"
        )

        st.progress(
            float(probs[i])
        )


st.caption(
    "※ 현재 모델은 4개 범주만 지원하며 "
    "프로젝트 데모용입니다."
)
