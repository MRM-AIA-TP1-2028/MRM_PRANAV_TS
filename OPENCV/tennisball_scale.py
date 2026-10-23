import math
from collections import deque

import cv2
import numpy as np

BLUE = (np.array([100, 120, 50]), np.array([130, 255, 255]))
BALL = (np.array([25, 80, 80]), np.array([45, 255, 255]))
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, WHITE = (0, 255, 0), (0, 255, 255), (255, 255, 255)
PALETTE = [(255, 0, 255), (0, 165, 255), (0, 0, 255), WHITE, (255, 255, 0)]
MAX_JUMP, MAX_MISSED, MIN_MOVE = 80, 15, 3
ARUCO_DICT, MARKER_CM, SCALE_HOLD = cv2.aruco.DICT_4X4_50, 5.0, 30


class Track:
    def __init__(self, tid, det):
        self.id, self.det, self.c = tid, det, det[2]
        self.anchor, self.dist, self.missed = self.c, 0.0, 0
        self.trail = deque([self.c], maxlen=64)

    @property
    def color(self):
        return PALETTE[(self.id - 1) % len(PALETTE)]


class State:
    def __init__(self):
        self.click, self.scale, self.lost = None, None, 0
        self.reset()

    def reset(self):
        self.blue, self.tracks, self.next_id = list(BLUE), [], 1


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return lambda gray: detector.detectMarkers(gray)[:2]
    params = cv2.aruco.DetectorParameters_create()
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


detect_markers = make_detector()


def update_scale(st, frame):
    corners, ids = detect_markers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        st.lost += 1
        if st.lost > SCALE_HOLD:
            st.scale = None
        return
    sides = [math.dist(c[i], c[(i + 1) % 4]) for c in (m.reshape(4, 2) for m in corners) for i in range(4)]
    new = MARKER_CM / (sum(sides) / len(sides))
    st.scale = new if st.scale is None else 0.8 * st.scale + 0.2 * new
    st.lost = 0
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)


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
            found.append((area, c, (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), cv2.minAreaRect(c)))
    return found


def find_ball(hsv):
    best, best_score = None, 0
    for c in blobs(hsv, BALL):
        area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
        (x, y), r = cv2.minEnclosingCircle(c)
        if area < 150 or perimeter == 0 or r == 0:
            continue
        circularity = 4 * math.pi * area / perimeter ** 2
        fill = area / (math.pi * r * r)
        if circularity >= 0.82 and fill >= 0.78 and area * circularity > best_score:
            best, best_score = ((int(x), int(y)), r, area, c), area * circularity
    return best


def track(st, dets):
    pairs = sorted((math.dist(t.c, d[2]), i, j)
                   for i, t in enumerate(st.tracks) for j, d in enumerate(dets))
    used_t, used_d = set(), set()
    for d, i, j in pairs:
        if d > MAX_JUMP:
            break
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        t = st.tracks[i]
        t.det, t.c, t.missed = dets[j], dets[j][2], 0
        t.trail.append(t.c)
        if math.dist(t.c, t.anchor) >= MIN_MOVE:
            t.dist += math.dist(t.c, t.anchor)
            t.anchor = t.c
    for i, t in enumerate(st.tracks):
        if i not in used_t:
            t.det, t.missed = None, t.missed + 1
    for j, d in enumerate(dets):
        if j not in used_d:
            st.tracks.append(Track(st.next_id, d))
            st.next_id += 1
    st.tracks = [t for t in st.tracks if t.missed <= MAX_MISSED]


def length(px, s):
    return f"{px * s:.1f} cm" if s else f"{px:.0f} px"


def area(px2, s):
    return f"{px2 * s * s:.1f} cm^2" if s else "no marker"


def side(d, low, high):
    return "aligned" if abs(d) <= 10 else high if d > 0 else low


def tag(frame, lines, pts, color, taken):
    h, w = frame.shape[:2]
    x0, y0 = pts.reshape(-1, 2).min(0)
    x1, y1 = pts.reshape(-1, 2).max(0)
    bw = max(cv2.getTextSize(t, FONT, 0.5, 1)[0][0] for t in lines) + 8
    bh = 18 * len(lines) + 6
    spots = [(x0, y0 - bh - 2), (x0, y1 + 2), (x1 + 4, y0), (x0 - bw - 4, y0)]
    rects = [(int(min(max(x, 0), w - bw)), int(min(max(y, 0), h - bh)), bw, bh) for x, y in spots]
    hit = lambda a, b: a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
    x, y, _, _ = rect = next((r for r in rects if not any(hit(r, t) for t in taken)), rects[0])
    taken.append(rect)
    roi = frame[y:y + bh, x:x + bw]
    cv2.addWeighted(np.zeros_like(roi), 0.6, roi, 0.4, 0, roi)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + 4, y + 17 + 18 * i), FONT, 0.5, color, 1, cv2.LINE_AA)


def info_window(tracks, ball, s):
    rows = sorted((t for t in tracks if t.det), key=lambda t: t.id)
    xs = [15, 65, 125, 190, 300, 460, 560, 670]
    heads = ["ID", "X px", "Y px", "Area", "Vs yellow ball", "Dist yellow", "Travelled", "Dist to other blue objects"]
    table = []
    for t in rows:
        if ball:
            dx, dy = t.c[0] - ball[0][0], t.c[1] - ball[0][1]
            vs = [f"{side(dx, 'LEFT', 'RIGHT')}, {side(dy, 'ABOVE', 'BELOW')}", length(math.hypot(dx, dy), s)]
        else:
            vs = ["no yellow ball", "-"]
        others = "  ".join(f"B{o.id}: {length(math.dist(t.c, o.c), s)}" for o in rows if o is not t) or "-"
        table.append([f"B{t.id}", str(t.c[0]), str(t.c[1]), area(t.det[0], s), *vs, length(t.dist, s), others])

    width = max(1000, xs[-1] + 20 + max([cv2.getTextSize(r[-1], FONT, 0.5, 1)[0][0] for r in table] + [200]))
    img = np.zeros((130 + 28 * max(len(rows), 1), width, 3), np.uint8)
    scale_text = f"SCALE   1 cm = {1 / s:.1f} px  (ArUco marker, {MARKER_CM:g} cm)" if s else "SCALE   no ArUco marker in view"
    cv2.putText(img, scale_text, (15, 26), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    if ball:
        text = (f"YELLOW BALL   center=({ball[0][0]}, {ball[0][1]})  radius={ball[1]:.1f} px  "
                f"area={area(ball[2], s)}")
    else:
        text = "YELLOW BALL   not detected"
    cv2.putText(img, text, (15, 52), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (10, 64), (width - 10, 64), WHITE, 1)
    for x, h in zip(xs, heads):
        cv2.putText(img, h, (x, 86), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (10, 96), (width - 10, 96), WHITE, 1)
    for k, row in enumerate(table):
        for x, v in zip(xs, row):
            cv2.putText(img, v, (x, 120 + 28 * k), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    if not rows:
        cv2.putText(img, "no blue objects", (15, 120), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return img


def process(frame, st):
    update_scale(st, frame)
    s = st.scale
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)

    if st.click:
        x, y = st.click
        hue, sat, val = np.median(hsv[max(y - 5, 0):y + 6, max(x - 5, 0):x + 6].reshape(-1, 3), axis=0).astype(int)
        st.blue = [np.array([max(hue - 12, 0), max(sat - 70, 40), max(val - 70, 30)]),
                   np.array([min(hue + 12, 179), 255, 255])]
        st.tracks, st.next_id, st.click = [], 1, None

    track(st, find_blue(hsv, st.blue))
    ball = find_ball(hsv)

    h, w = frame.shape[:2]
    taken = [(0, h - 45, 330, 45)]
    for t in st.tracks:
        if t.det:
            p = cv2.boxPoints(t.det[3])
            taken.append((p[:, 0].min(), p[:, 1].min(), np.ptp(p[:, 0]), np.ptp(p[:, 1])))
    if ball:
        r = ball[1]
        taken.append((ball[0][0] - r, ball[0][1] - r, 2 * r, 2 * r))
    if ball:
        cv2.drawContours(frame, [ball[3]], -1, GREEN, 2)
        cv2.circle(frame, ball[0], 3, GREEN, -1)
        tag(frame, [f"Ball ({ball[0][0]}, {ball[0][1]})", f"r={ball[1]:.1f} px  {area(ball[2], s)}"], ball[3], GREEN, taken)
    for t in st.tracks:
        if not t.det:
            continue
        box = cv2.boxPoints(t.det[3]).astype(np.int32)
        cv2.drawContours(frame, [t.det[1]], -1, t.color, 2)
        cv2.drawContours(frame, [box], -1, t.color, 1)
        cv2.circle(frame, t.c, 4, t.color, -1)
        if len(t.trail) > 1:
            cv2.polylines(frame, [np.array(t.trail, np.int32)], False, t.color, 2, cv2.LINE_AA)
        lines = [f"B{t.id}  {area(t.det[0], s)}", f"box {length(max(t.det[3][1]), s)} x {length(min(t.det[3][1]), s)}"]
        if ball:
            dx, dy = t.c[0] - ball[0][0], t.c[1] - ball[0][1]
            cv2.line(frame, t.c, ball[0], YELLOW, 1, cv2.LINE_AA)
            lines.append(f"{side(dx, 'LEFT', 'RIGHT')}, {side(dy, 'ABOVE', 'BELOW')} of ball  {length(math.hypot(dx, dy), s)}")
        tag(frame, lines, box, t.color, taken)
    note = f"1 cm = {1 / s:.1f} px" if s else "no ArUco marker: show one to get cm"
    cv2.putText(frame, note, (10, h - 30), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "click=select  r=reset  q=quit", (10, h - 10), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame, info_window(st.tracks, ball, s)


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
        frame, info = process(frame, st)
        cv2.imshow("Tracker", frame)
        cv2.imshow("Data", info)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            st.reset()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()