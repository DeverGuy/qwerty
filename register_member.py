# pyrefly: ignore [missing-import]
import cv2
import pickle
import os
# pyrefly: ignore [missing-import]
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

if os.path.exists("embeddings.pkl") and os.path.getsize("embeddings.pkl") > 0:
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
else:
    database = {}

name = input("Enter member name: ")

cap = cv2.VideoCapture(0)

while True:

    ret, raw_frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(raw_frame, 1)

    faces = app.get(frame)

    for face in faces:

        database[name] = face.embedding

        with open("embeddings.pkl", "wb") as f:
            pickle.dump(database, f)

        print(f"{name} registered successfully!")

        cap.release()
        cv2.destroyAllWindows()
        exit()

    cv2.imshow("Register Member", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()