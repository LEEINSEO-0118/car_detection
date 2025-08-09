from fastapi import FastAPI, File, UploadFile

from io import BytesIO
import onnxruntime as ort
import cv2
import numpy as np
import time

# Directory root
root = '/Users/toad/Documents/ToyProject/car_detection'

# FastAPI 앱 생성
app = FastAPI()

# Load model
model_path = f'{root}/code/best_pt/best.onnx'
model = ort.InferenceSession(model_path)


# 전처리 (onnx)
def preprocess_image(image):
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 640, 640)

    img = img / 255.0
    img = img.astype(np.float32)
    return img

# 후처리 (onnx)
def postprocess_image(image, predictions, conf):
    # 차원 축소
    predictions = np.array(predictions).squeeze()  # ex. [1][1][300][5] -> [300][6]

    # 원본 이미지 복사
    draw_img = image.copy()
    img_w, img_h = draw_img.shape[1], draw_img.shape[0]
    
    bbox_count = 0
    bbox_list = []
    scale_x = img_w / 640
    scale_y = img_h / 640
    for det in predictions:
        x1, y1, x2, y2, score, cls_id = det
        if score < conf:  # Confidence threshold
            continue
        # box count
        bbox_count += 1
        # 스케일 변환
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # 정수 좌표로 변환
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        bbox = [x1, y1, x2, y2, cls_id]
        bbox_list.append(bbox)

    return bbox_count, bbox_list

# byte to opencv
def bytes_to_cv2_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR 이미지 반환
    return img

# Fast API
@app.post('/detect')
async def detect_api(file: UploadFile = File(...)):
    # 비동기적으로 파일 읽기
    image_bytes = await file.read()

    try:
        # 이미지 열기
        origin_image = bytes_to_cv2_image(image_bytes)
        image = preprocess_image(origin_image.copy()) # RGB 변환 및 onnx 추론을 위한 전처리

        # 모델 추론
        start = time.time()
        predictions = model.run(None, {"images": image})
        end = time.time()

        # 후처리 (Bbox 갯수와 좌표 반환)
        bbox_count, bbox_list = postprocess_image(origin_image, predictions)

        # 추론시간
        infer_time = end - start  # 초 단위
        infer_ms = infer_time * 1000  # 밀리초 단위

        return {
            "Detection_counts": bbox_count,
            "Bbox_list": bbox_list,
            "Inference_time": infer_ms
        }
    except Exception as e:
        return {"error": "이미지 처리 실패", "details": str(e)}
    
# 이미지를 입력하면 탐지된 객체의 갯수, Bounding Box 좌표, 추론 소요 시간을 반환하는 API