#!/usr/bin/env python3
"""Live viewer + resistor colour-band reader for usb_cam_stream.ino.

Usage:
    python3 viewer.py /dev/cu.usbmodem1101     # live from the board
    python3 viewer.py saved_frame.png          # test on a saved image

Keys (in the video window):
    c  calibrate: measure a resistor whose colours you know, then type them
       in the terminal. Saved to calibration.json and exported to palette.h
       (the header the on-board ESP32 recognizer will use).
    o  region mode: drag a box around the resistor BODY with the mouse
       (for a fixed jig position); press o again to go back to auto-detect
    w  white balance: fill the view with white paper, then press w
    u  undo the last calibration      p  print measured band colours
    r  toggle resistor reading        m  debug window (mask + band profile)
    s  save current frame as PNG      q  quit

Setup assumptions: one resistor at a time, lying roughly flat on a plain
white / light-grey background, lit evenly (diffuse light, no hard shadows).
"""
import collections
import itertools
import datetime
import json
import struct
import sys
import time

import cv2
import numpy as np

# =====================================================================
# Tunables
# =====================================================================
# Foreground: a pixel is "resistor" if its colour (chroma) differs from the
# local background by more than FG_T. Brightness-only changes (shadows,
# vignetting) are ignored; neutral black/white bands are filled in by the
# hole-filling step. The background is estimated per pixel. Works on white or
# grey paper; for coloured backgrounds use region mode ('o').
FG_T = 10.0
MIN_AREA_FRAC = 0.005   # ignore blobs smaller than this fraction of the frame
BAND_T = 22.0           # band threshold on noisy images (upper limit)
BAND_T_MIN = 6.0        # band threshold on clean images (lower limit)
MIN_BAND_FRAC = 0.02    # min band width as a fraction of body length
END_TRIM = 0.06         # ignore this fraction at each end of the body (end caps)
ROI_END_TRIM = 0.02     # same, in region mode (you draw the box on the body)
DARK_W = 0.35           # weight of the "dark streak" cue for metallic (gold/silver)
                        # bands, which look like the body on average but have
                        # dark streaks that glare never produces
ALT_MARGIN = 15.0       # a band may be re-read as its 2nd/3rd closest colour if
                        # that is within this distance of the best match and it
                        # turns an invalid/non-standard code into a valid one
VOTE_FRAMES = 15        # temporal voting window (frames)
VOTE_NEEDED = 10        # votes needed to call a reading "stable"

CAL_FILE = "calibration.json"   # measured samples (created by calibrating)
HEADER_FILE = "palette.h"       # exported palette for the ESP32 sketch
CAL_FRAMES = 10                 # frames averaged per calibration capture

# Generic starting colours (RGB). Any colour you calibrate replaces its entry
# here with what your camera actually measured.
DEFAULT_PALETTE_RGB = {
    "black":  (25, 25, 25),
    "brown":  (95, 55, 35),
    "red":    (185, 35, 35),
    "orange": (225, 110, 30),
    "yellow": (225, 200, 40),
    "green":  (40, 135, 60),
    "blue":   (35, 75, 170),
    "violet": (120, 65, 150),
    "grey":   (125, 125, 125),
    "white":  (235, 235, 235),
    "gold":   (175, 145, 70),
    "silver": (185, 185, 190),
}

DIGIT = {"black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
         "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9}
MULT = {**{k: 10 ** v for k, v in DIGIT.items()}, "gold": 0.1, "silver": 0.01}
TOL = {"brown": "1%", "red": "2%", "green": "0.5%", "blue": "0.25%",
       "violet": "0.1%", "grey": "0.05%", "gold": "5%", "silver": "10%"}

E24 = {10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47,
       51, 56, 62, 68, 75, 82, 91}
E96 = {100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137,
       140, 143, 147, 150, 154, 158, 162, 165, 169, 174, 178, 182, 187, 191,
       196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267,
       274, 280, 287, 294, 301, 309, 316, 324, 332, 340, 348, 357, 365, 374,
       383, 392, 402, 412, 422, 432, 442, 453, 464, 475, 487, 499, 511, 523,
       536, 549, 562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
       750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976}
SERIES3 = E96 | {v * 10 for v in E24}   # valid 3-digit significands

LAB_W = np.array([0.6, 1.0, 1.0], np.float32)  # de-weight lightness (lighting varies)


def rgb_to_lab(rgb):
    px = np.uint8([[rgb[::-1]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def lab_to_rgb(lab):
    px = np.uint8([[np.clip(lab, 0, 255)]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0][::-1])


# =====================================================================
# Calibration
# =====================================================================
PALETTE_LAB = {}      # colour -> Lab reference used by classify()
CAL_SAMPLES = {}      # colour -> list of measured Lab samples
LAST_ADDED = []       # colours added by the last calibration (for undo)
SETTINGS = {"roi": None,        # [x, y, w, h] as fractions of the frame, or None
            "wb_gains": None}   # [b, g, r] software white-balance gains, or None
ALIASES = {"gray": "grey", "purple": "violet"}


def rebuild_palette():
    for k, rgb in DEFAULT_PALETTE_RGB.items():
        samples = CAL_SAMPLES.get(k)
        PALETTE_LAB[k] = (np.median(np.array(samples, np.float32), axis=0)
                          if samples else rgb_to_lab(rgb))


def load_calibration():
    try:
        with open(CAL_FILE) as f:
            data = json.load(f)
        CAL_SAMPLES.update(data.get("samples", {}))
        for k in SETTINGS:
            SETTINGS[k] = data.get(k)
        print(f"Loaded {CAL_FILE}")
    except FileNotFoundError:
        print(f"No {CAL_FILE} yet - using generic colours. Press 'c' to calibrate.")
    rebuild_palette()


def save_calibration():
    with open(CAL_FILE, "w") as f:
        json.dump({"samples": CAL_SAMPLES, **SETTINGS}, f, indent=1)
    write_header()


def write_header():
    names = list(DEFAULT_PALETTE_RGB)
    fmt = lambda t: "{" + ", ".join(f"{int(round(v)):3d}" for v in t) + "}"
    L = [f"// Generated by viewer.py on {datetime.datetime.now():%Y-%m-%d %H:%M}.",
         "// Resistor band colour palette, measured with the camera settings in the sketch.",
         "// If you change lighting or camera settings, recalibrate and regenerate this file.",
         "// LAB is OpenCV 8-bit Lab: L*255/100, a+128, b+128.",
         "#pragma once",
         "#include <stdint.h>",
         "",
         f"#define N_COLOURS {len(names)}",
         "enum Colour { " + ", ".join(f"C_{n.upper()}" for n in names) + " };",
         "static const char *const COLOUR_NAMES[N_COLOURS] = {"
         + ", ".join(f'"{n}"' for n in names) + "};",
         "static const uint8_t PALETTE_LAB[N_COLOURS][3] = {"]
    L += [f"  {fmt(PALETTE_LAB[n])},  // {n}" for n in names]
    L += ["};", "static const uint8_t PALETTE_RGB[N_COLOURS][3] = {"]
    L += [f"  {fmt(lab_to_rgb(PALETTE_LAB[n]))},  // {n}" for n in names]
    L += ["};", "// 1 = measured by calibration, 0 = generic default",
          "static const uint8_t PALETTE_CALIBRATED[N_COLOURS] = {"
          + ", ".join("1" if CAL_SAMPLES.get(n) else "0" for n in names) + "};",
          "",
          "// Recognizer tunables that were active during calibration",
          f"#define FG_T {FG_T}f",
          f"#define BAND_T {BAND_T}f",
          f"#define BAND_T_MIN {BAND_T_MIN}f",
          f"#define MIN_BAND_FRAC {MIN_BAND_FRAC}f",
          f"#define END_TRIM {END_TRIM}f",
          f"#define ROI_END_TRIM {ROI_END_TRIM}f",
          f"#define DARK_W {DARK_W}f",
          "",
          "// Software white balance, applied to every frame before recognition (B, G, R)"]
    g = SETTINGS["wb_gains"] or [1.0, 1.0, 1.0]
    L += [f"static const float WB_GAIN_BGR[3] = {{{g[0]:.4f}f, {g[1]:.4f}f, {g[2]:.4f}f}};",
          "// Fixed region around the resistor body, as fractions of the frame (0 = auto-detect)",
          f"#define USE_ROI {1 if SETTINGS['roi'] else 0}"]
    r = SETTINGS["roi"] or [0, 0, 0, 0]
    L += [f"static const float ROI_FRAC[4] = {{{r[0]:.4f}f, {r[1]:.4f}f, {r[2]:.4f}f, {r[3]:.4f}f}};  // x, y, w, h",
          "static const float LAB_W[3] = {" + ", ".join(f"{float(v):.3f}f" for v in LAB_W) + "};", ""]
    with open(HEADER_FILE, "w") as f:
        f.write("\n".join(L))


def print_coverage():
    done = [f"{k}({len(CAL_SAMPLES[k])})" for k in DEFAULT_PALETTE_RGB if CAL_SAMPLES.get(k)]
    todo = [k for k in DEFAULT_PALETTE_RGB if not CAL_SAMPLES.get(k)]
    print("  calibrated:", " ".join(done) or "none")
    print("  not yet:   ", " ".join(todo) or "none - all colours measured")
    names = list(DEFAULT_PALETTE_RGB)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm((PALETTE_LAB[a] - PALETTE_LAB[b]) * LAB_W))
            if d < BAND_T and (CAL_SAMPLES.get(a) or CAL_SAMPLES.get(b)):
                print(f"  warning: {a} and {b} look very similar to your camera "
                      f"(distance {d:.1f}) - expect confusion; improve lighting")


def apply_wb(img):
    g = SETTINGS["wb_gains"]
    if not g:
        return img
    return np.clip(img.astype(np.float32) * np.float32(g), 0, 255).astype(np.uint8)


def measure_wb(raw):
    """Set white-balance gains from a frame filled with white paper."""
    h, w = raw.shape[:2]
    m = np.median(raw[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3), axis=0).astype(np.float32)
    if m.min() < 30:
        print("White balance: centre of the image is too dark - is white paper in view?")
        return
    gains = (m.mean() / m).round(4).tolist()
    if CAL_SAMPLES:
        print("Changing white balance invalidates the colour calibration.")
        if ask("Clear calibration and use the new white balance? [y/N] ") != "y":
            print("white balance unchanged")
            return
        CAL_SAMPLES.clear()
        LAST_ADDED.clear()
        rebuild_palette()
    SETTINGS["wb_gains"] = gains
    save_calibration()
    print(f"White balance set. Paper measured RGB {tuple(int(v) for v in m[::-1])}, "
          f"gains R {gains[2]:.2f}  G {gains[1]:.2f}  B {gains[0]:.2f}")
    if max(gains) / min(gains) > 1.6:
        print("  (strong colour cast - a better WB_MODE in the sketch would reduce noise)")


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


def calibrate(frames, win):
    """Measure the bands over several frames, then ask what colours they are."""
    caps = []
    for _ in range(CAL_FRAMES * 3):
        img = next(frames)
        res = analyse(img)
        if res["bands"]:
            caps.append((img, res, np.array([b["lab"] for b in res["bands"]])))
        if len(caps) >= CAL_FRAMES:
            break
    if not caps:
        print("Calibration: no bands found. Check framing/lighting ('m' shows the mask).")
        return
    n = collections.Counter(len(c[2]) for c in caps).most_common(1)[0][0]
    caps = [c for c in caps if len(c[2]) == n]
    ref_img, ref_res, ref_labs = caps[0]
    runs = []
    for _, _, labs in caps:   # the strip can flip direction between frames
        if np.linalg.norm(labs[::-1] - ref_labs) < np.linalg.norm(labs - ref_labs):
            labs = labs[::-1]
        runs.append(labs)
    labs = np.median(np.array(runs), axis=0)

    cv2.imshow(win, draw(ref_img, ref_res, None, 0, status="CALIBRATING - see terminal"))
    cv2.waitKey(1)
    print(f"\nCalibration: found {n} bands (numbered 0..{n - 1} in the strip, left to right).")
    print("Type their colours in that order, e.g.  yellow violet red gold   (blank = cancel)")
    print("Use - for anything that is not a real band (e.g. a dark end cap).")
    names = [ALIASES.get(w, w) for w in ask("> ").split()]
    if not names:
        print("cancelled")
        return
    bad = [w for w in names if w not in DEFAULT_PALETTE_RGB and w != "-"]
    if bad or len(names) != n:
        print(f"Need exactly {n} names from: {' '.join(DEFAULT_PALETTE_RGB)}"
              + (f"  (unknown: {' '.join(bad)})" if bad else "") + " - cancelled")
        return

    def fit(seq):
        return sum(float(np.linalg.norm((PALETTE_LAB[k] - lab) * LAB_W))
                   for k, lab in zip(seq, labs) if k != "-")
    if names != names[::-1] and fit(names[::-1]) < 0.7 * fit(names):
        if ask("That looks like the reverse order of the strip. Reverse it? [Y/n] ") != "n":
            names = names[::-1]

    keep = [(k, lab) for k, lab in zip(names, labs) if k != "-"]
    names = [k for k, _ in keep]
    for name, lab in keep:
        prev = CAL_SAMPLES.get(name)
        if prev and np.linalg.norm((np.median(np.array(prev), axis=0) - lab) * LAB_W) > 2 * BAND_T:
            print(f"  warning: this '{name}' is very different from earlier '{name}' "
                  "samples - typo or lighting change? ('u' undoes)")
        CAL_SAMPLES.setdefault(name, []).append([round(float(v), 1) for v in lab])
    LAST_ADDED[:] = names
    rebuild_palette()
    save_calibration()
    print(f"Saved {CAL_FILE} and {HEADER_FILE}.")
    print_coverage()


def undo_last():
    if not LAST_ADDED:
        print("nothing to undo")
        return
    for name in LAST_ADDED:
        CAL_SAMPLES[name].pop()
        if not CAL_SAMPLES[name]:
            del CAL_SAMPLES[name]
    print(f"undid: {' '.join(LAST_ADDED)}")
    LAST_ADDED.clear()
    rebuild_palette()
    save_calibration()
    print_coverage()


def rank_colours(lab):
    """All palette colours sorted by distance: [(name, dist), ...]."""
    d = [(k, float(np.linalg.norm((v - lab) * LAB_W))) for k, v in PALETTE_LAB.items()]
    return sorted(d, key=lambda t: t[1])


def classify(lab):
    return rank_colours(lab)[0]


# =====================================================================
# Vision pipeline
# =====================================================================
def _ellipse(k):
    k = max(3, int(k) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _largest(mask):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lbl == i).astype(np.uint8) * 255


def find_resistor(img):
    """Return (minAreaRect of the resistor body, debug mask) or (None, mask)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Local background: a large median filter on a small copy removes the
    # resistor (a thin object) and keeps the slowly varying paper/lighting.
    small = cv2.resize(lab, (64, max(8, int(64 * h / w))), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 31), (w, h), interpolation=cv2.INTER_LINEAR)
    # Smooth first so sensor/JPEG noise does not light up scattered paper pixels
    d = cv2.GaussianBlur(lab.astype(np.float32), (5, 5), 0) - bg
    chroma = np.hypot(d[:, :, 1], d[:, :, 2])
    # Threshold adapts to the noise level (most of the frame is background)
    fg_t = max(FG_T, 4.0 * float(np.median(chroma)))
    mask = (chroma > fg_t).astype(np.uint8) * 255

    # 1) rough blob, only to measure how thick the body is
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(min(h, w) * 0.02))
    blob = _largest(mask)
    if blob is None:
        return None, mask
    radius = cv2.distanceTransform(blob, cv2.DIST_L2, 5).max()
    # 2) fill holes and bridge pale/white bands, THEN remove the leads, which
    #    are much thinner than the body (kernels scale with the body: any zoom)
    blob = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(radius * 1.5))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, _ellipse(radius))
    blob = _largest(blob)
    if blob is None or blob.sum() / 255 < MIN_AREA_FRAC * h * w:
        return None, mask
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.minAreaRect(max(contours, key=cv2.contourArea)), cv2.max(mask // 3, blob)


def rectify(img, rect):
    """Warp the rotated body rectangle into a horizontal strip (long axis = x)."""
    box = cv2.boxPoints(rect)
    d01 = np.linalg.norm(box[0] - box[1])
    d12 = np.linalg.norm(box[1] - box[2])
    if d01 >= d12:
        src, L, W = box[[0, 1, 2]], d01, d12
    else:
        src, L, W = box[[1, 2, 3]], d12, d01
    L, W = int(round(L)), int(round(W))
    if L < 20 or W < 5:
        return None
    dst = np.float32([[0, 0], [L, 0], [L, W]])
    M = cv2.getAffineTransform(np.float32(src), dst)
    return cv2.warpAffine(img, M, (L, W))


def read_bands(body, end_trim=END_TRIM):
    """Find colour bands along a rectified body. Returns (bands, body_lab, debug)."""
    h, w = body.shape[:2]
    x0, x1 = int(w * end_trim), int(w * (1 - end_trim))
    y0, y1 = int(h * 0.25), max(int(h * 0.75), int(h * 0.25) + 1)
    strip = cv2.cvtColor(body[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    prof = np.median(strip, axis=0)                     # 1-D colour profile (N, 3)
    prof = cv2.blur(prof.reshape(1, -1, 3), (3, 1)).reshape(-1, 3)
    n = len(prof)

    # Body colour = the colour most of the profile agrees with (mode seeking)
    d = np.linalg.norm((prof[:, None, :] - prof[None, :, :]) * LAB_W, axis=2)
    near = d < BAND_T * 0.6
    body_lab = np.median(prof[near[np.argmax(near.sum(axis=1))]], axis=0)

    raw_dist = np.linalg.norm((prof - body_lab) * LAB_W, axis=1)
    # Compare each point with the NEARBY body, not one global body colour:
    # subtracting a running baseline removes slow shading along the body
    # (dark end caps, uneven light) so faint bands still stand out.
    win = max(5, int(n * 0.25)) | 1

    def running(v, q):
        p = np.pad(v, win // 2, mode="edge")
        return np.percentile(np.lib.stride_tricks.sliding_window_view(p, win), q, axis=1)
    colour_sig = np.maximum(raw_dist - running(raw_dist, 30), 0)
    # Second cue: dark streaks. Take the darkest 15% of each column; metallic
    # bands dip well below the nearby body, glare only adds bright pixels.
    Lw = cv2.cvtColor(body[int(h * 0.15):max(int(h * 0.85), 1), x0:x1],
                      cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    dark = np.percentile(Lw, 15, axis=0)
    dark = cv2.blur(dark.reshape(1, -1), (3, 1)).ravel()
    # Reference = nearby body level, but never above the body's typical level,
    # so a body gap between two bright bands does not count as "dark".
    dark_ref = np.minimum(running(dark, 70), np.median(dark))
    dark_sig = np.maximum(dark_ref - dark, 0) * DARK_W

    # Each cue gets its own adaptive threshold from its own noise level (the
    # body is most of the profile), then they are combined as "how many times
    # over its threshold": 1.0 = just a band.
    def thresh(sig):
        return min(BAND_T, max(BAND_T_MIN, 6.0 * float(np.median(sig))))
    dist = np.maximum(colour_sig / thresh(colour_sig), dark_sig / thresh(dark_sig))
    thr = 1.0
    is_band = dist > thr
    min_w = max(2, int(n * MIN_BAND_FRAC))
    bands, i = [], 0
    while i < n:
        if not is_band[i]:
            i += 1
            continue
        j = i
        while j < n and is_band[j]:
            j += 1
        if j - i >= min_w:
            a, b = i + (j - i) // 5, j - (j - i) // 5        # inner part of the band
            lab = np.median(prof[a:max(b, a + 1)], axis=0)
            cands = rank_colours(lab)
            bands.append(dict(start=i + x0, end=j + x0, lab=lab, name=cands[0][0],
                              dist=cands[0][1], cands=cands[:3],
                              weak=float(dist[i:j].max()) < 2.5 * thr,
                              at_end=min(i, n - j) < 0.07 * w))
        i = j
    return bands, body_lab, dict(dist=dist, thr=thr, x0=x0, width=w)


def decode(names):
    """Colour names in reading order -> (ohms, tolerance, significand, ndigits)."""
    n = len(names)
    if n == 3:
        dig, mul, tol = names[:2], names[2], None
    elif n == 4:
        dig, mul, tol = names[:2], names[2], names[3]
    elif n in (5, 6):                     # 6th band = temp. coefficient, ignored
        dig, mul, tol = names[:3], names[3], names[4]
    else:
        return None
    if any(x not in DIGIT for x in dig) or mul not in MULT:
        return None
    if tol is not None and tol not in TOL:
        return None
    if DIGIT[dig[0]] == 0:
        return None
    sig = int("".join(str(DIGIT[x]) for x in dig))
    return round(sig * MULT[mul], 4), TOL.get(tol, "20%"), sig, len(dig)


def interpret(bands):
    """Pick the most plausible reading.

    Each band may be read as any colour within ALT_MARGIN of its best match;
    every combination is decoded in both directions and scored by colour fit
    plus how "resistor-like" the result is (valid code, standard E-series
    value, gold/silver tolerance band at the end, wider gap before it).
    """
    # A weak band right at either end of the body may be end-cap shading
    # rather than a real band: also try the reading without it (with a penalty).
    cap = [b.get("weak") and b.get("at_end") for b in bands]
    subsets = [(bands, 0.0)]
    if len(bands) > 3 and cap[0]:
        subsets.append((bands[1:], 6.0))
    if len(bands) > 3 and cap[-1]:
        subsets.append((bands[:-1], 6.0))
    if len(bands) > 4 and cap[0] and cap[-1]:
        subsets.append((bands[1:-1], 12.0))
    best = None
    for subset, penalty in subsets:
        r = _interpret(subset, penalty)
        if r and (best is None or r["cost"] < best["cost"]):
            best = r
    return best


def _interpret(bands, penalty):
    if not 3 <= len(bands) <= 6:
        return None
    gaps = [bands[k + 1]["start"] - bands[k]["end"] for k in range(len(bands) - 1)]
    options = [[c for c in b["cands"] if c[1] <= b["cands"][0][1] + ALT_MARGIN]
               for b in bands]
    raw = [b["name"] for b in bands]
    best = None
    for combo in itertools.product(*options):
        names = [c[0] for c in combo]
        fit = sum(c[1] for c in combo)
        for rev in (False, True):
            seq = names[::-1] if rev else names
            res = decode(seq)
            if res is None:
                continue
            ohms, tol, sig, ndig = res
            cost = fit + penalty
            if sig in (E24 if ndig == 2 else SERIES3):
                cost -= 15                                  # standard value
            if seq[-1] in ("gold", "silver"):
                cost -= 8                                   # tolerance band at the end
            if len(seq) == 3:
                cost += 6                                   # 3-band parts are rare
            tol_gap, first_gap = (gaps[0], gaps[-1]) if rev else (gaps[-1], gaps[0])
            if tol_gap > first_gap:
                cost -= 4                                   # wider gap before tolerance
            if best is None or cost < best["cost"]:
                best = dict(ohms=ohms, tol=tol, names=seq, cost=cost,
                            corrected=names != raw)
    return best


def fmt_ohms(r):
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if r >= div:
            return f"{r / div:g}{suf}"
    return f"{r:g}"


# =====================================================================
# Frame sources
# =====================================================================
def serial_frames(port):
    import serial
    MAGIC, MAX_FRAME = b"FRAM", 500_000
    ser = serial.Serial(port, 115200, timeout=2)
    print(f"Opened {port}, waiting for frames...")

    def read_exact(n):
        data = bytearray()
        while len(data) < n:
            chunk = ser.read(n - len(data))
            if not chunk:
                raise TimeoutError("No data from board")
            data += chunk
        return bytes(data)

    def sync_to_magic():
        window, junk = b"", bytearray()
        while True:
            b = ser.read(1)
            if not b:
                raise TimeoutError("No data from board")
            window = (window + b)[-4:]
            junk += b
            if window == MAGIC:
                text = bytes(junk[:-4])
                if text.strip():
                    print(text.decode(errors="replace").strip())
                return

    while True:
        try:
            sync_to_magic()
            (length,) = struct.unpack("<I", read_exact(4))
            if length == 0 or length > MAX_FRAME:
                continue
            jpg = read_exact(length)
        except TimeoutError as e:
            print(e)
            continue
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield img


def image_frames(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    while True:
        yield img.copy()


# =====================================================================
# Main loop
# =====================================================================
def analyse(img):
    """Full pipeline on one frame. Returns dict with everything for display."""
    out = dict(rect=None, body=None, bands=None, body_lab=None, reading=None, mask=None)
    roi = SETTINGS["roi"]
    if roi:                                   # fixed region: no detection needed
        H, W = img.shape[:2]
        x, y = int(roi[0] * W), int(roi[1] * H)
        w, h = max(1, int(roi[2] * W)), max(1, int(roi[3] * H))
        body = img[y:y + h, x:x + w]
        if h > w:                             # vertical jig: turn it horizontal
            body = cv2.rotate(body, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out["rect"] = ((x + w / 2, y + h / 2), (w, h), 0.0)
        trim = ROI_END_TRIM
    else:
        rect, out["mask"] = find_resistor(img)
        if rect is None:
            return out
        out["rect"] = rect
        body = rectify(img, rect)
        trim = END_TRIM
    if body is None or body.shape[1] < 20 or body.shape[0] < 5:
        return out
    out["body"] = body
    out["bands"], out["body_lab"], out["profile"] = read_bands(body, trim)
    out["reading"] = interpret(out["bands"])
    return out


def debug_view(res, width=480):
    """Mask + straightened body + colour-distance profile (for tuning)."""
    parts = []
    mask = res.get("mask")
    if mask is not None:
        m = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        m = cv2.resize(m, (width, int(m.shape[0] * width / m.shape[1])), interpolation=cv2.INTER_NEAREST)
        cv2.putText(m, "mask: bright = resistor body, dim = other foreground", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        parts.append(m)
    body, prof = res.get("body"), res.get("profile")
    if body is not None and prof is not None:
        parts.append(cv2.resize(body, (width, 50), interpolation=cv2.INTER_NEAREST))
        H = 120
        plot = np.full((H, width, 3), 25, np.uint8)
        d, thr, x0, bw = prof["dist"], prof["thr"], prof["x0"], prof["width"]
        top = max(float(d.max()), thr * 1.5, 1.0)
        xs = ((np.arange(len(d)) + x0) * width / bw).astype(int)
        ys = (H - 1 - d / top * (H - 10)).astype(int)
        cv2.polylines(plot, [np.stack([xs, ys], 1).astype(np.int32)], False, (255, 255, 255), 1)
        ty = int(H - 1 - thr / top * (H - 10))
        cv2.line(plot, (0, ty), (width, ty), (0, 0, 255), 1)
        for b in res.get("bands") or []:
            cv2.line(plot, (int(b["start"] * width / bw), H - 3), (int(b["end"] * width / bw), H - 3), (0, 255, 0), 3)
        cv2.putText(plot, "band signal: colour or dark-streak cue (red line = band threshold)",
                    (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        parts.append(plot)
    return np.vstack(parts) if parts else None


ROI_DRAG = {"on": False, "p0": None, "p1": None, "scale": 1}


def draw(img, res, stable, fps, status=None):
    vis = img.copy()
    if ROI_DRAG["on"] and ROI_DRAG["p0"] and ROI_DRAG["p1"]:
        cv2.rectangle(vis, ROI_DRAG["p0"], ROI_DRAG["p1"], (0, 200, 255), 1)
    if res.get("rect") is not None:
        cv2.drawContours(vis, [np.intp(cv2.boxPoints(res["rect"]))], 0, (0, 255, 0), 1)
    sc = 2 if vis.shape[1] < 500 else 1
    ROI_DRAG["scale"] = sc
    vis = cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    W = vis.shape[1]
    panel = np.full((110, W, 3), 30, np.uint8)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5

    body, bh = res.get("body"), 40
    if body is not None:
        strip = cv2.resize(body, (W, bh), interpolation=cv2.INTER_NEAREST)
        k = W / body.shape[1]
        for b in res["bands"] or []:
            x0, x1 = int(b["start"] * k), int(b["end"] * k)
            cv2.rectangle(strip, (x0, 0), (x1, bh - 1), (255, 255, 255), 1)
            idx = str(res["bands"].index(b))
            cv2.putText(strip, idx, (x0 + 2, 14), font, 0.45, (0, 0, 0), 3)
            cv2.putText(strip, idx, (x0 + 2, 14), font, 0.45, (255, 255, 255), 1)
        panel[0:bh] = strip
        names = " ".join(b["name"] for b in res["bands"] or []) or "no bands found"
        cv2.putText(panel, names, (5, bh + 18), font, fs, (200, 200, 200), 1)
    else:
        cv2.putText(panel, "no resistor found", (5, 58), font, fs, (200, 200, 200), 1)

    r = res.get("reading")
    now = f"{fmt_ohms(r['ohms'])} ohm +/-{r['tol']}" if r else "--"
    cv2.putText(panel, f"frame: {now}", (5, 80), font, fs, (200, 200, 200), 1)
    if status:
        cv2.putText(panel, status, (5, 102), font, 0.6, (0, 200, 255), 2)
    else:
        cv2.putText(panel, f"stable: {stable or '--'}", (5, 102), font, 0.6,
                    (80, 255, 80) if stable else (120, 120, 120), 2)
    # Focus meter: variance of the Laplacian in the centre of the frame.
    # Higher = sharper. Use it while turning the lens; only compare values
    # for the same scene and lighting.
    h, w = img.shape[:2]
    centre = cv2.cvtColor(img[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.COLOR_BGR2GRAY)
    focus = cv2.Laplacian(centre, cv2.CV_64F).var()
    mode = "DRAG A BOX ON THE BODY" if ROI_DRAG["on"] else ("region" if SETTINGS["roi"] else "auto")
    cv2.putText(vis, f"{fps} fps   focus {focus:.0f}   [{mode}]", (5, 18), font, fs, (0, 0, 255), 1)
    return np.vstack([vis, panel])


def on_mouse(event, x, y, flags, frame_shape):
    if not ROI_DRAG["on"]:
        return
    sc = ROI_DRAG["scale"]
    H, W = frame_shape[0]
    p = (min(max(x // sc, 0), W - 1), min(max(y // sc, 0), H - 1))
    if event == cv2.EVENT_LBUTTONDOWN:
        ROI_DRAG["p0"], ROI_DRAG["p1"] = p, p
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and ROI_DRAG["p0"]:
        ROI_DRAG["p1"] = p
    elif event == cv2.EVENT_LBUTTONUP and ROI_DRAG["p0"]:
        (x0, y0), (x1, y1) = ROI_DRAG["p0"], p
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        ROI_DRAG.update(on=False, p0=None, p1=None)
        if x1 - x0 < 10 or y1 - y0 < 4:
            print("region too small - press o and try again")
            return
        SETTINGS["roi"] = [round(x0 / W, 4), round(y0 / H, 4),
                           round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)]
        save_calibration()
        print(f"Region set: x {x0}-{x1}, y {y0}-{y1} (saved). Press o to clear it.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    live = not src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    frames = serial_frames(src) if live else image_frames(src)

    win = "ESP32S3-CAM"
    load_calibration()
    raw_frames = frames
    last_raw = {"img": None}

    def corrected():
        for f in raw_frames:
            last_raw["img"] = f
            yield apply_wb(f)
    frames = corrected()
    frame_shape = [(240, 320)]
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse, frame_shape)
    reading_on, show_mask = True, False
    votes = collections.deque(maxlen=VOTE_FRAMES)
    last_stable, fps, nframes, t0 = None, 0, 0, time.time()
    res = {}

    for img in frames:
        frame_shape[0] = img.shape[:2]
        nframes += 1
        if time.time() - t0 >= 1.0:
            fps, nframes, t0 = nframes, 0, time.time()

        stable = None
        if reading_on:
            res = analyse(img)
            r = res["reading"]
            votes.append(f"{fmt_ohms(r['ohms'])} ohm +/-{r['tol']}" if r else None)
            counts = collections.Counter(v for v in votes if v)
            if counts:
                top, cnt = counts.most_common(1)[0]
                stable = top if cnt >= VOTE_NEEDED else None
            if stable and stable != last_stable:
                print(f"STABLE READING: {stable}   bands: {' '.join(r['names']) if r else ''}")
            last_stable = stable
        else:
            res = {}

        cv2.imshow(win, draw(img, res, stable, fps))
        if show_mask:
            dbg = debug_view(res)
            if dbg is not None:
                cv2.imshow("debug", dbg)

        key = cv2.waitKey(1 if live else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            reading_on = not reading_on
            votes.clear()
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("debug")
        elif key == ord("s"):
            name = f"frame_{int(time.time())}.png"
            cv2.imwrite(name, img)
            print(f"saved {name}")
        elif key == ord("c"):
            calibrate(frames, win)
            votes.clear()
        elif key == ord("u"):
            undo_last()
        elif key == ord("o"):
            if SETTINGS["roi"] or ROI_DRAG["on"]:
                SETTINGS["roi"] = None
                ROI_DRAG.update(on=False, p0=None, p1=None)
                save_calibration()
                print("Region cleared - back to auto-detect.")
            else:
                ROI_DRAG["on"] = True
                print("Drag a box around the resistor BODY (from end cap to end cap, not the leads).")
            votes.clear()
        elif key == ord("w"):
            measure_wb(last_raw["img"])
            votes.clear()
        elif key == ord("p"):
            if res.get("bands"):
                print(f"body colour RGB {lab_to_rgb(res['body_lab'])}")
                for k, b in enumerate(res["bands"]):
                    alts = ", ".join(f"{n} {d:.0f}" for n, d in b["cands"])
                    print(f"  band {k}: RGB {lab_to_rgb(b['lab'])}  -> {alts}")
            else:
                print("no bands to print")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
