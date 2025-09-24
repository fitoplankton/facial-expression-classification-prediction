# utils.py
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA

# ===== OD-LBP =====
def pixel_value(image, row, col):
    h, w = image.shape
    if row in range(h) and col in range(w):
        return image[row, col]
    return 0

def variance(numbers):
    if len(numbers) < 2:
        return 1e-8
    mean = np.mean(numbers)
    squared_diff = [(x - mean) ** 2 for x in numbers]
    var = sum(squared_diff) / (len(numbers) - 1)
    return 1e-8 if var == 0.0 else var

def calculate_odlb(img_block):
    T1, T2, T3, T8, Tc, T4, T7, T6, T5 = img_block
    D = [
        [T1 - T2, T1 - Tc, T1 - T8],
        [T7 - T6, T7 - Tc, T7 - T8],
        [T5 - T4, T5 - Tc, T5 - T6],
        [T3 - T2, T3 - Tc, T3 - T4],
        [T2 - T3, T2 - Tc, T2 - T1],
        [T8 - T7, T8 - Tc, T8 - T1],
        [T6 - T5, T6 - Tc, T6 - T7],
        [T4 - T3, T4 - Tc, T4 - T5]
    ]
    A = [np.sum(d) / variance(d) for d in D]
    BPs = [[1 if n >= a else 0 for n in d] for d, a in zip(D, A)]
    CC = np.concatenate(BPs)
    binary_split = (CC[:8], CC[8:16], CC[16:])
    decimals = [sum(bit * (2 ** i) for i, bit in enumerate(code)) for code in binary_split]
    return decimals

def odlbp_image(img_gray):
    h, w = img_gray.shape
    img_gray = img_gray.astype(np.int16)
    odlbp1, odlbp2, odlbp3 = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            img_block = [
                pixel_value(img_gray, row - 1, col - 1),
                pixel_value(img_gray, row - 1, col),
                pixel_value(img_gray, row - 1, col + 1),
                pixel_value(img_gray, row, col - 1),
                pixel_value(img_gray, row, col),
                pixel_value(img_gray, row, col + 1),
                pixel_value(img_gray, row + 1, col - 1),
                pixel_value(img_gray, row + 1, col),
                pixel_value(img_gray, row + 1, col + 1)
            ]
            code = calculate_odlb(img_block)
            odlbp1[row, col], odlbp2[row, col], odlbp3[row, col] = code
    return odlbp1, odlbp2, odlbp3

def extract_histogram_from_image(image, total_row=3, total_column=3, bins_per_window=256):
    histogram = []
    h, w = image.shape
    win_h, win_w = h // total_row, w // total_column
    for row in range(total_row):
        for col in range(total_column):
            window = image[row*win_h:(row+1)*win_h, col*win_w:(col+1)*win_w]
            window_hist = np.histogram(window, bins=bins_per_window)[0]
            histogram.extend(window_hist)
    return np.array(histogram)

def odlbp_feature(gray_img):
    img1, img2, img3 = odlbp_image(gray_img)
    hist1 = extract_histogram_from_image(img1)
    hist2 = extract_histogram_from_image(img2)
    hist3 = extract_histogram_from_image(img3)
    return np.concatenate([hist1, hist2, hist3])

# ===== HOG =====
def extract_hog_feature(image, cell_size=(8, 8)):
    return hog(image, orientations=9, pixels_per_cell=cell_size, cells_per_block=(2, 2),
               block_norm='L2-Hys', visualize=False, channel_axis=2)

# ===== PCA =====
import pickle

# === Load PCA yang sudah disimpan ===
pca8 = pickle.load(open("new_models/pca_hog_8.pkl", "rb"))
pca16 = pickle.load(open("new_models/pca_hog_16.pkl", "rb"))
pca_odlbp = pickle.load(open("new_models/pca_odlbp.pkl", "rb"))

# ===== PCA Apply pakai model pickle =====
def apply_pca8x8(features):
    features = np.array(features).flatten()

    # Jika data ada label, buang kolom terakhir
    if features.shape[0] == pca8.n_features_in_ + 1:
        X = features[:-1]
    else:
        X = features

    # reshape sebelum PCA
    X = X.reshape(1, -1)
    reduced = pca8.transform(X)
    return reduced.flatten()


    # kalau ada label, gabungkan lagi
    if y is not None:
        return np.hstack((reduced.flatten(), y))
    return reduced.flatten()


def apply_pca16x16(features):
    features = np.array(features).reshape(1, -1)
    reduced = pca16.transform(features)
    return reduced.flatten()

def apply_pca_ODLBP(features):
    features = np.array(features).reshape(1, -1)
    reduced = pca_odlbp.transform(features)
    return reduced.flatten()

# ===== Main feature extraction =====
def extract_features(image_path, method):
    face_cascade_file = 'cascade_classifier/face-detect.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_file)
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 1.3, 1)

    for x, y, w, h in faces:
        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop wajah dari gambar RGB
        face_crop = rgb_img[y:y + h, x:x + w]
        face_crop_odlbp = gray_img[y:y + h, x:x + w]

        # Resize ke 256x256
        face_resized = cv2.resize(face_crop, (256, 256))    
        face_resized_odlbp = cv2.resize(face_crop_odlbp, (256, 256))    

    if method == "hog8x8":
        return extract_hog_feature(face_resized, cell_size=(8, 8))
    elif method == "hog16x16":
        return extract_hog_feature(face_resized, cell_size=(16, 16))
    elif method == "hog8x8_pca":
        hog8x8 = extract_hog_feature(face_resized, cell_size=(8, 8))
        return apply_pca8x8(hog8x8)
    elif method == "hog16x16_pca":
        hog16x16 = extract_hog_feature(face_resized, cell_size=(16, 16))
        return apply_pca16x16(hog16x16)
    elif method == "odlbp":
        return odlbp_feature(face_resized_odlbp)
    elif method == "odlbp_pca":
        odlbp = odlbp_feature(face_resized_odlbp)
        return apply_pca_ODLBP(odlbp)
    else:
        raise ValueError(f"Unknown method: {method}")