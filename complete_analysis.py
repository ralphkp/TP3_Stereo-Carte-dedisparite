import cv2
import matplotlib.pyplot as plt

def adjust_sift_threshold(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contrast_thresholds = [0.03, 0.04, 0.1]  # Ajustez ces valeurs en fonction des résultats
    fig, axes = plt.subplots(1, len(contrast_thresholds), figsize=(20, 10))

    for i, thresh in enumerate(contrast_thresholds):
        sift = cv2.SIFT_create(contrastThreshold=thresh)
        keypoints, _ = sift.detectAndCompute(img, None)
        keypoint_img = cv2.drawKeypoints(img, keypoints, None)
        axes[i].imshow(keypoint_img, cmap='gray')
        axes[i].set_title(f'Threshold: {thresh}, Keypoints: {len(keypoints)}')
        axes[i].axis('off')

    plt.show()

if __name__ == "__main__":
    adjust_sift_threshold('path_to_your_image.jpg')
