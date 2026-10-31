from ultralytics import YOLO
import cv2

# تحميل الموديل
model = YOLO("yolo26n.pt")

# فتح الكاميرا
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

print("YOLO Camera Started!")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # اكتشاف الأجسام
    results = model(frame)

    # رسم المربعات والكلاسات
    annotated_frame = results[0].plot()

    # عرض الكاميرا
    cv2.imshow("YOLO Live Object Detection", annotated_frame)

    # الخروج بزر Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()