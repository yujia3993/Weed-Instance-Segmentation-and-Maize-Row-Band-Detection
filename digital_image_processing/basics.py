# =============================================================================
# Digital Image Processing - Basics Notes
#
# Covers:
#   1. Image reading and color space conversion (BGR↔RGB, grayscale)
#   2. Brightness/contrast adjustment
#   3. Histogram equalization
#   4. Smoothing filters (mean, Gaussian, median)
#   5. Edge detection (Sobel, Canny parameter comparison)
#   6. Thresholding (fixed threshold, Otsu auto threshold)
#   7. Morphological operations (erosion, dilation, opening, closing)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------- Image path (replace with local path) ----------
IMAGE_PATH = r"C:\Users\Administrator\Desktop\flower.jpg"


# =============================================================================
# 1. Image Reading and Color Space Conversion
# =============================================================================

def demo_color_conversion(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Original shape: {img.shape}")
    print(f"Gray shape: {img_gray.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Gray Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return img, img_rgb, img_gray


# =============================================================================
# 2. Brightness / Contrast Adjustment + Histogram Equalization
# =============================================================================

def demo_brightness_contrast(img_gray: np.ndarray):
    brighter        = cv2.convertScaleAbs(img_gray, alpha=1.0, beta=40)
    darker          = cv2.convertScaleAbs(img_gray, alpha=1.0, beta=-40)
    higher_contrast = cv2.convertScaleAbs(img_gray, alpha=1.2, beta=0)
    lower_contrast  = cv2.convertScaleAbs(img_gray, alpha=0.3, beta=0)
    equalized       = cv2.equalizeHist(img_gray)

    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate([
        (img_gray,        "Original Gray"),
        (brighter,        "Brighter (+40)"),
        (darker,          "Darker (-40)"),
        (higher_contrast, "Higher Contrast (α=1.2)"),
        (lower_contrast,  "Lower Contrast (α=0.3)"),
        (equalized,       "Histogram Equalization"),
    ], 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Histogram comparison before/after equalization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(img_gray.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title("Histogram of Original Gray")
    plt.subplot(1, 2, 2)
    plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title("Histogram After Equalization")
    plt.tight_layout()
    plt.show()

    # Note: for images with many bright areas, simply increasing global contrast tends to
    # lose highlight detail; histogram equalization is adaptive but still struggles to
    # achieve ideal segmentation in complex natural images.


# =============================================================================
# 3. Smoothing Filters (with salt-and-pepper noise comparison)
# =============================================================================

def demo_smoothing(img_rgb: np.ndarray):
    mean_blur     = cv2.blur(img_rgb, (11, 11))
    gaussian_blur = cv2.GaussianBlur(img_rgb, (11, 11), 0)
    median_blur   = cv2.medianBlur(img_rgb, 11)

    plt.figure(figsize=(6, 8))
    for i, (image, title) in enumerate([
        (img_rgb,      "Original"),
        (mean_blur,    "Mean Blur"),
        (gaussian_blur,"Gaussian Blur"),
        (median_blur,  "Median Blur"),
    ], 1):
        plt.subplot(2, 2, i)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Salt-and-pepper noise comparison
    rng       = np.random.default_rng(42)
    noisy_img = img_rgb.copy()
    prob      = 0.03
    rnd       = rng.random(noisy_img.shape[:2])
    noisy_img[rnd < prob / 2]       = [0,   0,   0  ]
    noisy_img[rnd > 1 - prob / 2]   = [255, 255, 255]

    plt.figure(figsize=(12, 8))
    for i, (image, title) in enumerate([
        (img_rgb,                             "Original"),
        (noisy_img,                           "Salt-and-Pepper Noise"),
        (cv2.blur(noisy_img, (5, 5)),         "Mean Blur"),
        (cv2.GaussianBlur(noisy_img, (5,5),0),"Gaussian Blur"),
        (cv2.medianBlur(noisy_img, 5),        "Median Blur"),
    ], 1):
        plt.subplot(2, 3, i)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. Edge Detection (Sobel + Canny parameter tuning)
# =============================================================================

def demo_edge_detection(img_gray: np.ndarray):
    sobel_x     = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y     = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_comb  = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    canny       = cv2.Canny(img_gray, 100, 200)

    plt.figure(figsize=(12, 4))
    for i, (image, title) in enumerate([
        (img_gray,   "Gray"),
        (sobel_comb, "Sobel (x+y)"),
        (canny,      "Canny (100-200)"),
    ], 1):
        plt.subplot(1, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Canny threshold comparison
    plt.figure(figsize=(12, 4))
    for i, (low, high) in enumerate([(50, 150), (100, 200), (150, 250)], 1):
        edge = cv2.Canny(img_gray, low, high)
        plt.subplot(1, 3, i)
        plt.imshow(edge, cmap="gray")
        plt.title(f"Canny {low}-{high}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. Thresholding
# =============================================================================

def demo_thresholding(img_gray: np.ndarray):
    _, binary_80  = cv2.threshold(img_gray,  80, 255, cv2.THRESH_BINARY)
    _, binary_120 = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    _, binary_160 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
    ret_otsu, binary_otsu = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure(figsize=(12, 8))
    for i, (image, title) in enumerate([
        (img_gray,    "Gray Image"),
        (binary_80,   "Threshold = 80"),
        (binary_120,  "Threshold = 120"),
        (binary_160,  "Threshold = 160"),
        (binary_otsu, f"Otsu T={ret_otsu:.0f}"),
    ], 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Fixed thresholds are highly sensitive to the chosen value; Otsu automatically
    # computes the threshold adaptively, but still struggles with complex natural images.

    return binary_otsu


# =============================================================================
# 6. Morphological Operations
# =============================================================================

def demo_morphology(binary_otsu: np.ndarray):
    kernel   = np.ones((5, 5), np.uint8)
    erosion  = cv2.erode(binary_otsu, kernel, iterations=1)
    dilation = cv2.dilate(binary_otsu, kernel, iterations=1)
    opening  = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN,  kernel)
    closing  = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(12, 8))
    for i, (image, title) in enumerate([
        (binary_otsu, "Otsu Binary"),
        (erosion,     "Erosion"),
        (dilation,    "Dilation"),
        (opening,     "Opening"),
        (closing,     "Closing"),
    ], 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Erosion shrinks foreground regions; dilation expands them.
    # Opening removes small white noise and smooths boundaries;
    # closing fills small black holes to make foreground more connected.


# =============================================================================
# Main entry: demonstrate all processing steps in order
# =============================================================================

if __name__ == "__main__":
    img, img_rgb, img_gray = demo_color_conversion(IMAGE_PATH)
    demo_brightness_contrast(img_gray)
    demo_smoothing(img_rgb)
    demo_edge_detection(img_gray)
    binary_otsu = demo_thresholding(img_gray)
    demo_morphology(binary_otsu)
