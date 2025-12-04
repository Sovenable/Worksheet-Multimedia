import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, detrend, find_peaks, windows
import sys
import warnings

warnings.filterwarnings('ignore')

BUFFER_SECONDS = 60
WINDOW_SECONDS = 20
UPDATE_INTERVAL = 1.0

ROI_SELECTION = "forehead"
if len(sys.argv) > 1 and sys.argv[1].lower() in ['forehead', 'cheeks']:
    ROI_SELECTION = sys.argv[1].lower()

WIDTH_RATIO = 0.18
HEIGHT_PX = 70
PAD_Y = -8

BANDPASS_LOW = 0.67
BANDPASS_HIGH = 4.0

BPM_SMOOTH_WINDOW = 6
BPM_HOLD_SECONDS = 5.0

MOVING_AVG_FRAMES = 25
MIN_SIGNAL_SAMPLES = 60

SPEC_PANEL_RATIO = 0.28

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_first_valid_landmark(landmarks, candidates, img_w, img_h):
    for idx in candidates:
        try:
            lm = landmarks[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            return x, y
        except Exception:
            continue
    return int(img_w * 0.5), int(img_h * 0.18)

def landmarks_to_forehead_box(landmarks, img_w, img_h, width_ratio=WIDTH_RATIO, height_px=HEIGHT_PX, pad_y=PAD_Y):
    left_candidates = [70, 63, 105]
    right_candidates = [300, 293, 334]
    lx, ly = get_first_valid_landmark(landmarks, left_candidates, img_w, img_h)
    rx, ry = get_first_valid_landmark(landmarks, right_candidates, img_w, img_h)
    center_x = int((lx + rx) / 2)
    brows_mean_y = int((ly + ry) / 2)
    bottom_y = int(brows_mean_y + pad_y)
    top_y = bottom_y - height_px
    dist_alis = abs(rx - lx)
    width_default = int(max(int(img_w * width_ratio), int(dist_alis * 0.9)))
    half_w = width_default // 2
    left_x = center_x - half_w
    right_x = center_x + half_w
    left_x = int(np.clip(left_x, 0, img_w - 1))
    right_x = int(np.clip(right_x, 0, img_w - 1))
    top_y = int(np.clip(top_y, 0, img_h - 1))
    bottom_y = int(np.clip(bottom_y, 0, img_h - 1))
    pts = np.array([[left_x, top_y],
                    [right_x, top_y],
                    [right_x, bottom_y],
                    [left_x, bottom_y]], dtype=np.int32)
    pts = cv2.convexHull(pts)
    return pts

def landmarks_to_cheeks_box(landmarks, img_w, img_h):
    left_cheek_candidates = [226, 113, 50, 2]
    right_cheek_candidates = [446, 343, 280, 398]
    rois = []
    lx, ly = get_first_valid_landmark(landmarks, left_cheek_candidates, img_w, img_h)
    box_size = int(img_h * 0.10)
    left_x = int(max(0, lx - box_size // 2.8))
    right_x = int(min(img_w - 1, lx + box_size // 2.8))
    top_y = int(max(0, ly + box_size // 4))
    bottom_y = int(min(img_h - 1, ly + box_size // 0.9))
    pts = np.array([[left_x, top_y],
                    [right_x, top_y],
                    [right_x, bottom_y],
                    [left_x, bottom_y]], dtype=np.int32)
    rois.append(cv2.convexHull(pts))
    rx, ry = get_first_valid_landmark(landmarks, right_cheek_candidates, img_w, img_h)
    left_x = int(max(0, rx - box_size // 2.8))
    right_x = int(min(img_w - 1, rx + box_size // 2.8))
    top_y = int(max(0, ry + box_size // 4))
    bottom_y = int(min(img_h - 1, ry + box_size // 0.9))
    pts = np.array([[left_x, top_y],
                    [right_x, top_y],
                    [right_x, bottom_y],
                    [left_x, bottom_y]], dtype=np.int32)
    rois.append(cv2.convexHull(pts))
    return rois

def get_roi_boxes(landmarks, img_w, img_h, roi_selection='forehead'):
    boxes = []
    names = []
    if roi_selection == 'forehead':
        boxes.append(landmarks_to_forehead_box(landmarks, img_w, img_h))
        names.append('Forehead')
    elif roi_selection == 'cheeks':
        cheek_boxes = landmarks_to_cheeks_box(landmarks, img_w, img_h)
        boxes.extend(cheek_boxes)
        names.extend(['Left Cheek', 'Right Cheek'])
    else:
        boxes.append(landmarks_to_forehead_box(landmarks, img_w, img_h))
        names.append('Forehead')
    return boxes, names

def mean_color_in_mask(frame_bgr, mask):
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return None
    b, g, r = cv2.split(frame_bgr)
    return float(r[mask_bool].mean()), float(g[mask_bool].mean()), float(b[mask_bool].mean())

def resample_signal_uniform(times, signal, fs_target, window_seconds):
    if len(times) < 3:
        return None, None
    t_end = times[-1]
    t_start = t_end - window_seconds
    times_np = np.array(times)
    sig_np = np.array(signal)
    mask = times_np >= t_start
    times_in = times_np[mask]
    sig_in = sig_np[mask]
    if len(times_in) < 3:
        return None, None
    N = max(int(window_seconds * fs_target), 4)
    t_uniform = np.linspace(times_in[0], times_in[-1], N)
    sig_uniform = np.interp(t_uniform, times_in, sig_in)
    return t_uniform, sig_uniform

def detrend_signal(sig, method='moving_avg', moving_avg_frames=MOVING_AVG_FRAMES):
    if method == 'moving_avg':
        k = max(3, moving_avg_frames)
        if len(sig) < k:
            k = max(3, int(len(sig) / 4))
        mov = np.convolve(sig, np.ones(k) / k, mode='same')
        return sig - mov
    else:
        return detrend(sig, type=method)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, lowcut, highcut, fs, order=4):
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, sig)
        return y
    except Exception:
        return sig

def extract_signal_pos(times, signal_r, signal_g, signal_b):
    r = np.array(signal_r)
    g = np.array(signal_g)
    b = np.array(signal_b)
    if len(r) < 3:
        return None
    r_n = r / (np.mean(r) + 1e-8)
    g_n = g / (np.mean(g) + 1e-8)
    b_n = b / (np.mean(b) + 1e-8)
    X = np.vstack([r_n, g_n, b_n])
    S = np.array([[0, 1, -1],
                  [-2, 1, 1]])
    P = S.dot(X)
    h = P[0, :] - P[1, :]
    return h

def estimate_bpm_from_signal(sig, fs, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH):
    sig = np.array(sig)
    if len(sig) < 4:
        return None, None, None
    sig = sig - np.mean(sig)
    win = windows.hamming(len(sig))
    sig_win = sig * win
    sig_bp = bandpass_filter(sig_win, lowcut, highcut, fs, order=4)
    N = len(sig_bp)
    fft = np.abs(np.fft.rfft(sig_bp))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    idx_band = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    if idx_band.size < 3:
        return None, fft, freqs
    freqs_band = freqs[idx_band]
    fft_band = fft[idx_band]
    if np.max(fft_band) > 0:
        fft_norm = fft_band / np.max(fft_band)
    else:
        fft_norm = fft_band
    peaks, props = find_peaks(fft_norm, height=0.02, prominence=0.08)
    if peaks.size == 0:
        peak_idx = np.argmax(fft_norm)
        peak_freq = freqs_band[peak_idx]
        bpm = peak_freq * 60.0
        if bpm < 30 or bpm > 220:
            return None, fft, freqs
        return bpm, fft, freqs
    scores = props.get('prominences', np.ones_like(peaks)) * props.get('peak_heights', np.ones_like(peaks))
    best = np.argmax(scores)
    peak_idx = peaks[best]
    peak_freq = freqs_band[peak_idx]
    bpm = peak_freq * 60.0
    if bpm < 30 or bpm > 220:
        return None, fft, freqs
    return bpm, fft, freqs

def run(roi_selection=ROI_SELECTION):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0
    fs_target = fps

    buffer_len = int(BUFFER_SECONDS * max(fps, fs_target))
    times = deque(maxlen=buffer_len)
    signal_r = deque(maxlen=buffer_len)
    signal_g = deque(maxlen=buffer_len)
    signal_b = deque(maxlen=buffer_len)

    bpm_history = deque(maxlen=BPM_SMOOTH_WINDOW)
    last_valid_bpm = None
    last_valid_time = 0.0
    last_fft = None
    last_freqs = None

    last_update = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            freqs = None
            fft = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                boxes, names = get_roi_boxes(landmarks, w, h, roi_selection)
                total_r = total_g = total_b = 0.0
                valid = 0
                for box in boxes:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, box, 255)
                    mean_rgb = mean_color_in_mask(frame, mask)
                    if mean_rgb is not None:
                        r_mean, g_mean, b_mean = mean_rgb
                        total_r += r_mean
                        total_g += g_mean
                        total_b += b_mean
                        valid += 1
                        cv2.polylines(frame, [box], True, (0,255,0), 2)
                if valid > 0:
                    r_mean = total_r / valid
                    g_mean = total_g / valid
                    b_mean = total_b / valid
                    t = time.time()
                    times.append(t)
                    signal_r.append(r_mean)
                    signal_g.append(g_mean)
                    signal_b.append(b_mean)

                    cv2.putText(frame, f"G: {g_mean:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(frame, f"ROI: {roi_selection}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    cv2.putText(frame, "No valid ROI", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            now = time.time()
            bpm_est = None
            peak_freq_hz = None

            if (now - last_update) >= UPDATE_INTERVAL and len(times) >= MIN_SIGNAL_SAMPLES:
                last_update = now
                t_uniform, sig_green = resample_signal_uniform(list(times), list(signal_g), fs_target, WINDOW_SECONDS)
                _, sig_r = resample_signal_uniform(list(times), list(signal_r), fs_target, WINDOW_SECONDS)
                _, sig_b = resample_signal_uniform(list(times), list(signal_b), fs_target, WINDOW_SECONDS)

                if t_uniform is not None:
                    pos_sig = None
                    try:
                        pos_sig = extract_signal_pos(list(times), list(signal_r), list(signal_g), list(signal_b))
                        if pos_sig is not None:
                            _, pos_res = resample_signal_uniform(list(times), list(pos_sig), fs_target, WINDOW_SECONDS)
                        else:
                            pos_res = None
                    except Exception:
                        pos_res = None

                    if pos_res is not None and len(pos_res) >= 4:
                        sig_choice = pos_res
                    else:
                        sig_choice = sig_green

                    sig_detrended = detrend_signal(sig_choice, method='moving_avg', moving_avg_frames=MOVING_AVG_FRAMES)

                    bpm_est, fft, freqs = estimate_bpm_from_signal(sig_detrended, fs_target)
                    if fft is not None and freqs is not None:
                        last_fft = fft
                        last_freqs = freqs
                        idxb = np.where((freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH))[0]
                        if idxb.size > 1:
                            freqs_band = freqs[idxb]
                            fft_band = fft[idxb]
                            if np.max(fft_band) > 0:
                                peak_local = np.argmax(fft_band)
                                peak_freq_hz = freqs_band[peak_local]

                    if bpm_est is not None:
                        bpm_history.append(bpm_est)
                        last_valid_bpm = bpm_est
                        last_valid_time = now

                    sig_len = len(sig_choice) if sig_choice is not None else 0
                    sig_std = float(np.std(sig_choice)) if sig_len>0 else 0.0
                    bpm_print = f"{bpm_est:.1f}" if bpm_est is not None else "--"
                    peak_print = f"{peak_freq_hz:.2f}" if peak_freq_hz is not None else "--"
                    print(f"[DBG] samples={sig_len} std={sig_std:.4f} bpm={bpm_print} peak={peak_print}")

            spec_h = int(h * SPEC_PANEL_RATIO)
            spec_img = np.zeros((spec_h, w, 3), dtype=np.uint8) + 28

            draw_fft = last_fft
            draw_freqs = last_freqs
            if draw_fft is not None and draw_freqs is not None:
                idxb = np.where((draw_freqs >= BANDPASS_LOW) & (draw_freqs <= BANDPASS_HIGH))[0]
                if idxb.size > 1:
                    freqs_band = draw_freqs[idxb]
                    fft_band = draw_fft[idxb]
                    mag = fft_band - np.min(fft_band)
                    if np.max(mag) > 0:
                        mag = mag / np.max(mag)
                    xs = np.linspace(0, w-1, len(mag)).astype(np.int32)
                    ys = (spec_h - 40) - (mag * (spec_h - 70)).astype(np.int32)
                    pts = np.vstack([xs, ys]).T.reshape(-1,1,2)
                    cv2.polylines(spec_img, [pts], False, (0,180,255), 2)
                    peak_local = np.argmax(mag)
                    px = int(xs[peak_local]); py = int(ys[peak_local])
                    peak_freq_hz = freqs_band[peak_local]; peak_bpm = peak_freq_hz*60.0
                    cv2.circle(spec_img, (px, py), 4, (0,255,0), -1)
                    cv2.putText(spec_img, f"Peak: {peak_freq_hz:.2f}Hz ({peak_bpm:.0f} BPM)", (px+8, py-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)

            bpm_display = "--"
            if last_valid_bpm is not None and (time.time() - last_valid_time) <= BPM_HOLD_SECONDS:
                bpm_display = f"{last_valid_bpm:.1f}"
            elif len(bpm_history) > 0:
                bpm_display = f"{np.mean(bpm_history):.1f}"

            cv2.putText(spec_img, f"BPM: {bpm_display}", (10, spec_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,220,0), 2, cv2.LINE_AA)

            try:
                combined = cv2.vconcat([frame, spec_img])
            except Exception:
                spec_img_resized = cv2.resize(spec_img, (w, spec_h))
                combined = cv2.vconcat([frame, spec_img_resized])

            cv2.imshow("rPPG - Camera + BPM (Tekan 'q' untuk keluar)", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    time.sleep(0.6)
    run(roi_selection=ROI_SELECTION)
