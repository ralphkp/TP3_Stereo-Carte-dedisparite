import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_map(img1, img2):
    # Conversion en niveaux de gris si nécessaire
    if len(img1.shape) > 2:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) > 2:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # Paramètres pour StereoSGBM
    window_size = 5
    min_disp = 0
    num_disp = 16*5  # doit être divisible par 16
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = window_size,
                                   P1 = 8 * 3 * window_size ** 2,
                                   P2 = 32 * 3 * window_size ** 2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 15,
                                   speckleWindowSize = 0,
                                   speckleRange = 2,
                                   preFilterCap = 63,
                                   mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    # Calcul de la carte de disparité
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    return disparity

def post_process_disparity(disparity):
    # Appliquer un filtre médian pour réduire le bruit
    filtered_disparity = cv2.medianBlur(disparity, 5)
    # Normaliser pour améliorer la visualisation
    disparity_visual = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_visual

def test_inversion(img1, img2):
    disparity_original = compute_disparity_map(img1, img2)
    disparity_visual_original = post_process_disparity(disparity_original)

    disparity_inverted = compute_disparity_map(img2, img1)
    disparity_visual_inverted = post_process_disparity(disparity_inverted)

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(disparity_visual_original, cmap='jet')
    plt.title('Carte de Disparité - Ordre Original')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(disparity_visual_inverted, cmap='jet')
    plt.title('Carte de Disparité - Images Inversées')
    plt.colorbar()
    plt.show()

def main():
    img1 = cv2.imread('im1.jpeg')  # Mettez le chemin correct
    img2 = cv2.imread('im2.jpg')  # Mettez le chemin correct

    if img1 is None or img2 is None:
        print("Erreur : vérifiez les chemins des images.")
        return

    test_inversion(img1, img2)

if __name__ == "__main__":
    main()
