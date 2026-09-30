"""Auto-capture cards from a webcam.

Hold a card still under the camera. When a rectangle with text on it stays put for a
moment, the card is cropped, flattened and saved. Take it away before the next one.
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

WINDOW = "card scanner"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", help="output folder (default: scans/<date_time>)")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, default=1920, help="requested camera width")
    p.add_argument("--height", type=int, default=1080, help="requested camera height")
    p.add_argument("--hold", type=float, default=0.8, help="seconds the card must stay still")
    p.add_argument("--min-area", type=float, default=0.08, help="min card size as a fraction of the frame")
    p.add_argument("--min-text", type=float, default=0.008, help="min edge density inside the card to count as text")
    p.add_argument("--raw", action="store_true", help="also save the full uncropped frame")
    p.add_argument("--focus", type=int, help="lock manual focus at this value (turns autofocus off)")
    p.add_argument("--min-sharp", type=float, default=0, help="min sharpness to auto-capture (0 = off)")
    p.add_argument("--no-auto", action="store_true", help="start with auto-capture off (t turns it on)")
    return p.parse_args()


def order_corners(pts):
    # Top-left, top-right, bottom-right, bottom-left
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def to_quad(contour):
    # Fit 4 corners to the convex hull, None if it is not rectangle-like
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            break
    else:
        return None
    quad_area = cv2.contourArea(approx)
    if quad_area <= 0 or cv2.contourArea(contour) / quad_area < 0.85:
        return None
    # Reject very thin shapes
    (_, _), (w, h), _ = cv2.minAreaRect(approx)
    if min(w, h) / max(w, h) < 0.3:
        return None
    return approx


def find_cards(frame, min_area):
    # Rectangle-like contours, biggest first. Runs on a half-size copy for speed.
    scale = 960 / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    scale = min(scale, 1.0)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    fh, fw = gray.shape
    frame_area = fh * fw
    margin = 3
    quads = []
    for lo, hi, k in ((50, 150, 3), (20, 60, 5), (10, 40, 5)):
        edges = cv2.Canny(gray, lo, hi)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not min_area * frame_area <= area <= 0.9 * frame_area:
                continue
            quad = to_quad(c)
            if quad is None:
                continue
            # Skip shapes cut off by the frame edge
            xs, ys = quad[:, 0, 0], quad[:, 0, 1]
            if xs.min() > margin and ys.min() > margin and xs.max() < fw - margin and ys.max() < fh - margin:
                quads.append((cv2.contourArea(quad), order_corners(quad) / scale))
    quads.sort(key=lambda q: -q[0])
    return [q for _, q in quads]


def find_card(frame, min_area, min_text):
    # Biggest rectangle with text on it, returns (corners, crop, is_card)
    quads = find_cards(frame, min_area)
    for corners in quads[:8]:
        card = warp(frame, corners)
        if text_density(card) >= min_text:
            return corners, card, True
    if quads:
        return quads[0], warp(frame, quads[0]), False
    return None, None, False


def warp(frame, corners):
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, m, (w, h))


def text_density(card):
    # Edge density of the inner area, ignoring the card border
    h, w = card.shape[:2]
    inner = card[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    if inner.size == 0:
        return 0.0
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    return float(np.count_nonzero(edges)) / edges.size


def sharpness(card):
    # Variance of the Laplacian, higher is sharper
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def next_index(folder):
    nums = [int(m.group(1)) for p in folder.glob("card_*.jpg") if (m := re.match(r"card_(\d+)\.jpg$", p.name))]
    return max(nums, default=0) + 1


def main():
    args = parse_args()
    here = Path(__file__).parent
    folder = Path(args.out) if args.out else here / "scans" / time.strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    n = next_index(folder)
    saved = []

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Focus
    autofocus = args.focus is None
    focus = args.focus if args.focus is not None else int(cap.get(cv2.CAP_PROP_FOCUS))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus)

    print(f"Camera {args.camera} at {fw}x{fh}, saving to {folder.resolve()}")
    print("Keys: space = capture now, t = toggle auto-capture, u = undo last, q/esc = quit")
    print("Focus: a = toggle autofocus, [ ] = manual focus -/+, c = camera settings dialog")

    still_since = None
    last_corners = None
    armed = True  # False after a capture until the card is taken away
    auto_on = not args.no_auto
    gone_frames = 0
    flash_until = 0.0
    move_tol = 0.015 * max(fw, fh)
    best = None  # (sharpness, card, frame) of the sharpest frame while holding still

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        now = time.time()
        corners, card, has_text = find_card(frame, args.min_area, args.min_text)
        sharp = sharpness(card) if card is not None else 0.0

        # Stillness check, keeping the sharpest frame of the hold
        if has_text and last_corners is not None and np.abs(corners - last_corners).max() < move_tol:
            still_since = still_since or now
        else:
            still_since = now if has_text else None
            best = None
        last_corners = corners if has_text else None
        if has_text and (best is None or sharp > best[0]):
            best = (sharp, card, frame)

        # Re-arm once the card has been gone for a few frames
        if has_text:
            gone_frames = 0
        else:
            gone_frames += 1
            if gone_frames >= 8:
                armed = True

        key = cv2.waitKey(1) & 0xFF
        manual = key == ord(" ")
        auto = (auto_on and armed and has_text and still_since is not None and now - still_since >= args.hold
                and best is not None and best[0] >= args.min_sharp)

        if manual or auto:
            if auto:
                shot_sharp, shot, shot_frame = best
            else:
                shot_sharp, shot, shot_frame = sharp, card if card is not None else frame, frame
            path = folder / f"card_{n:03d}.jpg"
            cv2.imwrite(str(path), shot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files = [path]
            if args.raw:
                raw = folder / f"card_{n:03d}_raw.jpg"
                cv2.imwrite(str(raw), shot_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                files.append(raw)
            saved.append(files)
            print(f"Saved {path.name} (sharpness {shot_sharp:.0f})")
            n += 1
            armed = False
            still_since = None
            best = None
            flash_until = now + 0.3
        elif key == ord("t"):
            auto_on = not auto_on
            still_since = None
            best = None
            print(f"Auto-capture {'on' if auto_on else 'off'}")
        elif key == ord("a"):
            autofocus = not autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            if not autofocus:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
            print(f"Autofocus {'on' if autofocus else f'off, focus {focus}'}")
        elif key in (ord("["), ord("]")):
            # Nudging focus switches to manual
            autofocus = False
            focus = max(0, min(255, focus + (5 if key == ord("]") else -5)))
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            print(f"Manual focus {focus}")
        elif key == ord("c"):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        elif key == ord("u") and saved:
            for f in saved.pop():
                f.unlink(missing_ok=True)
            n -= 1
            print(f"Removed card_{n:03d}")
        elif key in (ord("q"), 27):
            break

        # Preview
        view = frame.copy()
        if corners is not None:
            color = (0, 200, 0) if has_text else (0, 165, 255)
            if has_text and not armed:
                color = (200, 200, 0)
            cv2.polylines(view, [corners.astype(np.int32)], True, color, 3)
            if auto_on and has_text and armed and still_since is not None:
                frac = min(1.0, (now - still_since) / args.hold)
                cv2.rectangle(view, (20, fh - 40), (20 + int(300 * frac), fh - 20), (0, 200, 0), -1)
        if now < flash_until:
            view = cv2.addWeighted(view, 0.5, np.full_like(view, 255), 0.5, 0)
        if not armed:
            status = "saved - remove card"
        elif has_text:
            status = "hold still" if auto_on else "space to capture"
        elif corners is not None:
            status = "no text found"
        else:
            status = "show a card"
        focus_text = "auto" if autofocus else str(focus)
        lines = [f"{len(saved)} saved | next card_{n:03d} | {status}",
                 f"auto {'on' if auto_on else 'off'} | focus {focus_text} | sharpness {sharp:.0f}"]
        for i, text in enumerate(lines):
            y = 40 + i * 40
            cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        scale = min(1.0, 1280 / fw)
        if scale < 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.imshow(WINDOW, view)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {len(saved)} cards saved to {folder.resolve()}")


if __name__ == "__main__":
    main()
