import cv2
import os
import numpy as np

import time

from insightface.app import FaceAnalysis


app = FaceAnalysis(name='buffalo_s')
app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)



known_faces = {}

for file in os.listdir("faces"):

    path = os.path.join("faces", file)

    img = cv2.imread(path)

    faces = app.get(img)

    if len(faces) == 0:
        continue

    name = os.path.splitext(file)[0]
    
    known_faces[name] = faces[0].embedding

print("Database loaded:")
print(known_faces.keys())



def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )



cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_count = 0
while True:

    ret, frame = cap.read()

    if not ret:
        break
    small_frame = cv2.resize(
    frame,
    (320, 240)
)
    start = time.time() 
    frame_count += 1
    
    if frame_count % 3 == 0:
        faces = app.get(small_frame)
    print("AI:", time.time() - start)

    for face in faces:

        embedding = face.embedding

        best_name = "Unknown"
        best_score = 0

        for name, known_embedding in known_faces.items():

            score = cosine_similarity(
                embedding,
                known_embedding
            )

            if score > best_score:

                best_score = score
                best_name = name

        if best_score < 0.4:
            best_name = "Unknown"

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{best_name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

