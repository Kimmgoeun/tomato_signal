import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import zipfile


# =========================================================
# 페이지 설정
# =========================================================

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


CLASS_NAMES = [
    "정상",
    "초기역병",
    "잎곰팡이병",
    "황화잎말림바이러스"
]


# =========================================================
# 분할 모델 다시 합치기
# =========================================================

# streamlit_app.py가 실제로 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# GitHub에 올린 model.part00, model.part01 ...
model_parts = sorted(
    BASE_DIR.glob("model.part*")
)

if len(model_parts) == 0:
    st.error(
        "❌ 모델 조각(model.part00 등)을 찾을 수 없습니다."
    )
    st.stop()


# 합쳐진 모델은 임시폴더에 저장
MODEL_PATH = Path(
    "/tmp/tomato_4class_clean_v3_stage2_best.keras"
)

expected_size = sum(
    p.stat().st_size
    for p in model_parts
)


# 기존 임시 파일이 없거나 크기가 다르면 새로 합치기
if (
    not MODEL_PATH.exists()
    or MODEL_PATH.stat().st_size != expected_size
):

    with open(MODEL_PATH, "wb") as output_file:

        for part in model_parts:

            with open(part, "rb") as input_file:
                output_file.write(
                    input_file.read()
                )


# =========================================================
# 모델 파일 검증
# =========================================================

if MODEL_PATH.stat().st_size != expected_size:

    st.error(
        "❌ 모델 조각을 합치는 과정에서 "
        "파일 크기가 일치하지 않습니다."
    )

    st.stop()


# .keras는 내부적으로 ZIP 형식
if not zipfile.is_zipfile(MODEL_PATH):

    st.error(
        "❌ 합쳐진 모델 파일이 정상적인 "
        ".keras 파일이 아닙니다."
    )

    st.write(
        "발견한 모델 조각 수:",
        len(model_parts)
    )

    st.write(
        "모델 조각:",
        [p.name for p in model_parts]
    )

    st.write(
        "합쳐진 크기:",
        round(
            MODEL_PATH.stat().st_size
            / (1024 ** 2),
            2
        ),
        "MB"
    )

    st.stop()


# =========================================================
# 모델 로드
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )


model = load_model()


# =========================================================
# 이미지 업로드
# =========================================================

uploaded = st.file_uploader(
    "토마토 잎 이미지를 업로드하세요",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
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


    # -----------------------------------------
    # 모델 입력 변환
    # -----------------------------------------

    resized = image.resize(
        (224, 224)
    )

    x = np.asarray(
        resized,
        dtype=np.float32
    )

    x = np.expand_dims(
        x,
        axis=0
    )


    # -----------------------------------------
    # 예측
    # -----------------------------------------

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


    # -----------------------------------------
    # Confidence threshold
    # -----------------------------------------

    if max_prob < 0.60:

        st.warning(
            "⚠️ 지원 범위 외 증상이거나 "
            "판별 신뢰도가 낮습니다."
        )

        st.write(
            f"가장 가까운 예측: "
            f"**{CLASS_NAMES[pred_idx]}** "
            f"({max_prob * 100:.1f}%)"
        )

    else:

        st.success(
            f"### 예측 결과: "
            f"{CLASS_NAMES[pred_idx]}"
        )

        st.write(
            f"신뢰도: "
            f"**{max_prob * 100:.1f}%**"
        )


    # -----------------------------------------
    # 전체 확률
    # -----------------------------------------

    st.subheader(
        "클래스별 예측 확률"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        probability = float(
            probs[i]
        )

        st.write(
            f"{name}: "
            f"{probability * 100:.1f}%"
        )

        st.progress(
            probability
        )


st.caption(
    "※ 현재 모델은 정상, 초기역병, 잎곰팡이병, "
    "황화잎말림바이러스 4개 범주만 지원하며 "
    "프로젝트 데모용입니다."
)
