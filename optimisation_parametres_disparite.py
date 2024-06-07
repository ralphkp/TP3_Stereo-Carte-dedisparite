import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_image_texture(image):
    """Estime la texture de l'image en calculant l'écart-type des intensités des pixels."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray_image)

def adjust_stereo_parameters(texture_level):
    """Ajuste les paramètres de StereoSGBM basés sur le niveau de texture estimé de l'image."""
    if texture_level < 20:
        # Image avec peu de texture
        num_disparities = 16 * 12  # Moins de disparités pour éviter le bruit
        block_size = 5  # Taille de bloc plus petite
    else:
        # Image avec beaucoup de texture
        num_disparities = 16 * 16
        block_size = 9
    return num_disparities, block_size

def compute_disparity_map(img1, img2, num_disparities, block_size):
    """Calcule la carte de disparité en utilisant StereoSGBM."""
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=num_disparities,
                                   blockSize=block_size,
                                   P1=8 * 3 * block_size ** 2,
                                   P2=32 * 3 * block_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=15,
                                   speckleWindowSize=0,
                                   speckleRange=2,
                                   preFilterCap=63,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
    return disparity

def post_process_disparity(disparity):
    """Applique un filtre médian pour réduire le bruit dans la carte de disparité."""
    filtered_disparity = cv2.medianBlur(disparity, 5)
    disparity_visual = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_visual

def main():
    img1 = cv2.imread('im1.jpeg')  # Remplacer par le chemin de votre image gauche
    img2 = cv2.imread('im2.jpg') # Remplacer par le chemin de votre image droite

    if img1 is None or img2 is None:
        print("Erreur : vérifiez les chemins des images.")
        return

    texture_level = estimate_image_texture(img1)
    num_disparities, block_size = adjust_stereo_parameters(texture_level)

    disparity = compute_disparity_map(img1, img2, num_disparities, block_size)
    disparity_visual = post_process_disparity(disparity)

    # Afficher la carte de disparité
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity_visual, cmap='jet')
    plt.colorbar()
    plt.title('Carte de Disparité Optimisée')
    plt.show()

if __name__ == "__main__":
    main()
