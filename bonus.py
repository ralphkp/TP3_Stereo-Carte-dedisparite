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
    # Appliquer un filtre bilatéral pour lisser tout en préservant les bords
    bilateral_filtered_disparity = cv2.bilateralFilter(filtered_disparity, 9, 75, 75)
    # Normaliser pour améliorer la visualisation
    disparity_visual = cv2.normalize(bilateral_filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_visual

def main():
    img1 = cv2.imread('im1.jpeg')  # Remplacer par le chemin de votre image gauche
    img2 = cv2.imread('im2.jpg') # Remplacer par le chemin de votre image droite

    if img1 is None or img2 is None:
        print("Erreur : vérifiez les chemins des images.")
        return

    disparity = compute_disparity_map(img1, img2)
    disparity_visual = post_process_disparity(disparity)

    # Afficher la carte de disparité
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity_visual, cmap='jet')
    plt.colorbar()
    plt.title('Carte de Disparité avec Filtre Bilatéral')
    plt.show()

if __name__ == "__main__":
    main()