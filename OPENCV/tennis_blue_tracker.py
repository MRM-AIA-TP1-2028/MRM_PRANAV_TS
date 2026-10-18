import math
from collections import deque

import cv2
import numpy as np

BLUE = (np.array([100, 120, 50]), np.array([130, 255, 255]))
BALL = (np.array([25, 80, 80]), np.array([45, 255, 255]))
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAGENTA, GREEN, YELLOW = (255, 0, 255), (0, 255, 0), (0, 255, 255)


class State:
    def __init__(self):
        self.click = None
        self.reset()

    def reset(self):
        self.blue = list(BLUE)
        self.trail = deque(maxlen=64)


def blobs(hsv, rng):
    mask = cv2.inRange(hsv, *rng)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def find_blue(hsv, rng):
    found = []
    for c in blobs(hsv, rng):
        area = cv2.contourArea(c)
        if area >= 800:
            m = cv2.moments(c)
            found.append((area, cv2.boundingRect(c), (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))))
    return sorted(found, reverse=True)


def find_ball(hsv):
    best = None
    for c in blobs(hsv, BALL):
        area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
        (x, y), r = cv2.minEnclosingCircle(c)
        if area < 150 or perimeter == 0 or r == 0:
            continue
        circularity = 4 * math.pi * area / perimeter ** 2
        fill = area / (math.pi * r * r)
        if circularity >= 0.82 and fill >= 0.78 and (best is None or area * circularity > best[3]):
            best = ((int(x), int(y)), r, area, area * circularity)
    return best


def side(d, low, high):
    return "aligned" if abs(d) <= 10 else high if d > 0 else low


def label(frame, lines, x, y, color):
    h, w = frame.shape[:2]
    box_w = max(cv2.getTextSize(t, FONT, 0.5, 1)[0][0] for t in lines) + 8
    box_h = 20 * len(lines) + 4
    x, y = int(min(max(x, 0), w - box_w)), int(min(max(y, 0), h - box_h))
    roi = frame[y:y + box_h, x:x + box_w]
    cv2.addWeighted(np.zeros_like(roi), 0.6, roi, 0.4, 0, roi)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + 4, y + 16 + 20 * i), FONT, 0.5, color, 1, cv2.LINE_AA)
    return box_h


def process(frame, st):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)

    if st.click:
        x, y = st.click
        hue, sat, val = np.median(hsv[max(y - 5, 0):y + 6, max(x - 5, 0):x + 6].reshape(-1, 3), axis=0).astype(int)
        st.blue = [np.array([max(hue - 12, 0), max(sat - 70, 40), max(val - 70, 30)]),
                   np.array([min(hue + 12, 179), 255, 255])]
        st.trail.clear()
        st.click = None

    blues, ball = find_blue(hsv, st.blue), find_ball(hsv)
    rects = [b[1] for b in blues]
    if ball:
        rects.append((ball[0][0] - int(ball[1]), ball[0][1] - int(ball[1]), 2 * int(ball[1]), 2 * int(ball[1])))
    overlap = lambda x0: sum(max(0, min(x + bw, x0 + 240) - max(x, x0)) * bh for x, _, bw, bh in rects)
    panel_x = w if overlap(w - 240) <= overlap(0) else 10
    panel_y = 10

    if ball:
        (cx, cy), r, area, _ = ball
        cv2.circle(frame, (cx, cy), int(r), GREEN, 2)
        cv2.circle(frame, (cx, cy), 3, GREEN, -1)
        label(frame, ["Ball"], cx - 16, cy - int(r) - 22, GREEN)
        lines = ["TENNIS BALL", f"Center: ({cx}, {cy})", f"Radius: {r:.1f} px", f"Area: {area:.0f} px^2"]
    else:
        lines = ["TENNIS BALL", "not detected"]
    panel_y += label(frame, lines, panel_x, panel_y, GREEN) + 6

    if blues:
        st.trail.append(blues[0][2])
    else:
        st.trail.clear()

    for i, (area, (x, y, bw, bh), c) in enumerate(blues, 1):
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), MAGENTA, 2)
        cv2.circle(frame, c, 4, MAGENTA, -1)
        label(frame, [f"B{i}"], x, y - 22, MAGENTA)
        lines = [f"BLUE OBJECT B{i}", f"Area: {area:.0f} px^2", f"BBox: x={x} y={y} w={bw} h={bh}"]
        if ball:
            dx, dy = c[0] - ball[0][0], c[1] - ball[0][1]
            cv2.line(frame, c, ball[0], YELLOW, 1, cv2.LINE_AA)
            lines += [f"vs ball: {side(dx, 'LEFT', 'RIGHT')}, {side(dy, 'ABOVE', 'BELOW')}",
                      f"dx={dx:+d} dy={dy:+d} dist={math.hypot(dx, dy):.0f}px"]
        else:
            lines.append("vs ball: no ball in view")
        if panel_y + 20 * len(lines) < h - 30:
            panel_y += label(frame, lines, panel_x, panel_y, MAGENTA) + 6

    if len(st.trail) > 1:
        cv2.polylines(frame, [np.array(st.trail, np.int32)], False, MAGENTA, 2, cv2.LINE_AA)
    cv2.putText(frame, "click=select  r=reset  q=quit", (10, h - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    st = State()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            st.click = (x, y)

    cv2.namedWindow("Tracker")
    cv2.setMouseCallback("Tracker", on_mouse)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Tracker", process(frame, st))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            st.reset()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()