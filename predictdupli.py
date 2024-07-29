import os
from ultralytics import YOLO
import cv2
import numpy as np

#OTHER YOUTUBE REFERENCE
def point_side(x1, y1, x2, y2, x, y):
    v_x = x2 - x1
    v_y = y2 - y1
    w_x = x - x1
    w_y = y - y1

    cross_product = (v_x * w_y) - (v_y * w_x)

    if cross_product > 0:
        #return "Left"
        return 0
    elif cross_product < 0:
        #return "Right"
        return 1
    else:
        #return "On the line"
        return -1

# result = point_side(x1, y1, x2, y2, x, y)
# print(f"The point ({x}, {y}) is on the {result} side of the line.")

level1=[246,227,274,228,578,229,604,228]
level2=[219,288,251,289,601,289,633,289]
level3=[205,320,238,321,613,319,645,319]
level4=[187,358,224,359,629,359,666,359]
level5=[143,453,187,453,668,454,712,454]
level6=[133,476,179,475,677,476,723,477]

VIDEOS_DIR = os.path.join('..', 'Videos')
county=0
video_path = os.path.join(VIDEOS_DIR, '12.mp4')
video_path_out = '{}_finalout17.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('..', 'best2.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

lastcx=0
lastcy=0
threshold = 0.1

last_x1, last_x2, last_y1, last_y2 = None, None, None, None
trig=0
while ret:
    results = model(frame)[0]
    cv2.line(frame, (level1[0], level1[1]), (level1[6], level1[7]), thickness=3,color=(0, 255, 255))  # color changes when it passes
    cv2.line(frame, (level1[0], level1[1]), (level6[0], level6[1]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level6[0], level6[1]), (level6[6], level6[7]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level6[6], level6[7]), (level1[6], level1[7]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level1[2], level1[3]), (level6[2], level6[3]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level1[4], level1[5]), (level6[4], level6[5]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level2[0], level2[1]), (level2[6], level2[7]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level3[0], level3[1]), (level3[6], level3[7]), thickness=3, color=(0, 50, 200))
    cv2.line(frame, (level4[0], level4[1]), (level4[6], level4[7]), thickness=3, color=(0, 255, 255))
    cv2.line(frame, (level5[0], level5[1]), (level5[6], level5[7]), thickness=3, color=(0, 255, 255))
    if trig==0:
        cv2.line(frame, (level1[0], level1[1]), (level1[6], level1[7]), thickness=3,color=(250,50,0))  # color changes when it passes
        cv2.line(frame, (level2[0], level2[1]), (level2[6], level2[7]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level1[0], level1[1]), (level3[0], level3[1]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level1[2], level1[3]), (level3[2], level3[3]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level1[4], level1[5]), (level3[4], level3[5]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level1[6], level1[7]), (level3[6], level3[7]), thickness=3, color=(250,50,0))
    else:
        cv2.line(frame, (level6[0], level6[1]), (level6[6], level6[7]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level5[0], level5[1]), (level5[6], level5[7]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level4[0], level4[1]), (level4[6], level4[7]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level3[0], level3[1]), (level6[0], level6[1]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level3[2], level3[3]), (level6[2], level6[3]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level3[4], level3[5]), (level6[4], level6[5]), thickness=3, color=(250,50,0))
        cv2.line(frame, (level3[6], level3[7]), (level6[6], level6[7]), thickness=3, color=(250,50,0))

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        lastcx=int(cx)
        lastcy=int(cy)
        if cy<=288 :
            trig=0
        else:
            trig=1
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            last_x1, last_y1, last_x2, last_y2 = int(x1), int(y1), int(x2), int(y2)
        county=county+1

    out.write(frame)
    ret, frame = cap.read()

# Access the last values outside the loop
if last_x1 is not None:
    print("Last x1:", last_x1)
    print("Last y1:", last_y1)
    print("Last x2:", last_x2)
    print("Last y2:", last_y2)
    print("lastcx",lastcx)
    print("lastcy",lastcy)
    print("count",county)

# Add a blank frame with text at the end
blank_frame = 255 * np.ones((H, W, 3), dtype=np.uint8)  # White blank frame
rightside=point_side(level1[4],level1[5],level6[4],level6[5],lastcx,lastcy)
leftside=point_side(level1[2],level1[3],level6[2],level6[3],lastcx,lastcy)

point_allocated=-1
#point_allocated=0 point to top player
#point_allocated=1 point to bottom player


if level3[3]<lastcy<level6[3]:
    if rightside==0 and leftside==1:
        point_allocated=0
    else:
        point_allocated=1
elif level1[3]<lastcy<level3[3]:
    if rightside==0 and leftside==1:
        point_allocated=1
    else:
        point_allocated=0
#write condition here to even allocate points outside the court

if point_allocated==0:
    cv2.putText(blank_frame, "player in the top got the point", (int(W / 4), int(H / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                cv2.LINE_AA)
else:
    cv2.putText(blank_frame, "player in the bottom got the point", (int(W / 4), int(H / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                cv2.LINE_AA)
#
# cv2.putText(blank_frame, "player point", (int(W / 4), int(H / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
#                 cv2.LINE_AA)

#cv2.putText(blank_frame, "Your Text Here", (int(W/4), int(H/2)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
for _ in range(int(cap.get(cv2.CAP_PROP_FPS)) * 5):  # Add the blank frame for 5 seconds
    out.write(blank_frame)

cap.release()
out.release()
cv2.destroyAllWindows()


