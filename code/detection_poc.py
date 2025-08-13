import streamlit as st
import requests
import cv2

root = '/Users/toad/Documents/ToyProject/car_detection'
url = 'http://127.0.0.1:8000/detect'

################## Functions

## Box 시각화 함수
def postprocess_image(image, predictions):
    # RGB 변환 되지 않은 이미지를 받아야 함
    # 원본 이미지 복사
    draw_img = image.copy()

    # 클래스 이름 (필요시 수정)
    class_names = ["car"]

    # api 반환 값 분리
    coords, infer_time = predictions
    for det in coords:
        # 좌표 가져오기
        x1, y1, x2, y2, score, cls_id = det

        # 박스 그리기
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # BGR기준

        # 클래스 & 점수 표시
        label = f"{class_names[cls_id]} {score:.2f}"
        cv2.putText(draw_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    text = f'Inference Time : {infer_time:0.4}ms'
    cv2.putText(draw_img, text, (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # BGR → RGB 변환
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    return draw_img

## Requests 함수
def detection_request(url, image, conf):
    response = requests.post(url, files=image, data=conf)
    return response.status_code, response.json()

## Traffic Evaluation
def traffic_eval(count):
    if 0 <= count <=5:
        return 'Free Flow'
    elif 6 <= count <=15:
        return 'Moderate'
    elif 16 <= count:
        return 'Heavy'

################## View
st.title('Traffic Monitoring System')

st.header('Real Time Traffic Monitoring', divider='gray')

# 이미지 결과 화면, 설명 화면 분할
image_col, text_col = st.columns([0.6,0.4], vertical_alignment="center")
image_path = f'{root}/test.png'
conf = 0.4
# 이미지 파일 to binary
with open(image_path, 'rb') as f: # 파일을 binary로 읽기
    files = {'file' : (image_path, f, 'image/png')}
    data = {'confidence' : conf}
    response_code, reponse_json = detection_request(url, files, data) # api 요청
bbox_cnt, coords, infer_time = reponse_json.values() # 3가지 값 분리
# 결과 이미지 시각화
with image_col:
    image = cv2.imread(image_path)
    post_image = postprocess_image(image, [coords, infer_time])
    st.image(post_image)
# 결과 이미지 해석
with text_col:
    st.subheader("Detection Summary")
    st.metric(label="Detected Vehicles", value=bbox_cnt)
    st.metric(label="Traffic Evaluation", value=traffic_eval(bbox_cnt))
    st.metric(label="Detection Speed (ms)", value=round(infer_time,2))














