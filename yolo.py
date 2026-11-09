
import cv2
from ultralytics import YOLO

class DogCatCls:
    # YOLOv8 모델 로드
    def __init__(self):
        self.detect_model = YOLO('yolov8l.pt')
        self.cls_model = YOLO('yolov8x-cls.pt')

    # 이미지 처리 및 품종 예측
    def process_image(self, image, target_animal="dog", conf_threshold=0.25):
        result_img = image.copy()
        detections = []

        # 객체 탐지
        results = self.detect_model.predict(
            image, conf=conf_threshold, verbose=False
        )

        # 탐지된 객체 중 target_animal에 해당하는 것만 필터링
        for r in results:
            for box in r.boxes:
                animal = self.detect_model.names[int(box.cls[0])].lower()

                if animal != target_animal:
                    continue

                # bbox 좌표 추출 및 이미지 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det_conf = float(box.conf[0])
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # 품종 분류 및 Top-5 예측
                probs = self.cls_model.predict(crop, verbose=False)[0].probs
                top5 = []

                for idx, conf in zip(probs.top5, probs.top5conf):
                    top5.append({
                        "breed": self.cls_model.names[int(idx)].replace('_', ' ').title(),
                        "confidence": float(conf)
                    })

                breed = top5[0]["breed"]
                cls_conf = top5[0]["confidence"]

                # BBox 및 라벨 표시
                color = (0, 255, 0) if animal == "dog" else (255, 105, 180)
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

                label = f"{breed} ({cls_conf:.2f})"
                cv2.putText(result_img, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 분석 결과 저장
                detections.append({
                    "animal": animal,
                    "breed": breed,
                    "confidence": cls_conf,
                    "detection_confidence": det_conf,
                    "bbox": [x1, y1, x2, y2],
                    "top5": top5
                })

        return result_img, detections
