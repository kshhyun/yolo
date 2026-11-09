
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from yolo import DogCatCls

st.set_page_config(page_title="YOLOv8 품종 예측", layout="wide")
st.title("YOLOv8 개/고양이 품종 예측")

# 입력 설정
target = st.sidebar.radio("예측 대상", ["dog", "cat"])
conf = st.sidebar.slider("탐지 신뢰도(conf) 기준", 0.1, 0.9, 0.25, 0.05)
file = st.sidebar.file_uploader("이미지 선택", type=["jpg", "png", "jpeg"])

# 이미지 업로드
if file:
    # 이미지 전처리
    image = Image.open(file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 품종 분석
    with st.spinner("품종 분석 중..."):
        model = DogCatCls()
        result_img, detections = model.process_image(img_bgr, target, conf)

    # 원본 및 결과 이미지
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("예측 결과")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # 객체 상세 정보
    st.markdown("---")
    st.subheader("객체 상세 정보")

    if detections:
        for i, obj in enumerate(detections, 1):
            st.write(
                f"**[{i}] Class:** `{obj['animal']}` | "
                f"**품종:** `{obj['breed']}` | "
                f"**품종 신뢰도:** `{obj['confidence']:.4f}` | "
                f"**탐지 신뢰도:** `{obj['detection_confidence']:.4f}` | "
                f"**BBox 좌표:** `{obj['bbox']}`"
            )
            # Top-5 결과 출력
            st.markdown("---")
            st.subheader("Top-5 분류 예측 결과")
            for rank, pred in enumerate(obj["top5"], 1):
                st.write(f"{rank}. {pred['breed']} ({pred['confidence']:.4f})")
    else:
        st.warning("탐지 실패 : 다른 이미지를 업로드 혹은 탐지 신뢰도(conf) 조정")
else:
    st.info("이미지 업로드하기")
