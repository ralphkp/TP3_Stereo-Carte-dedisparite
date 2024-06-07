import cv2

def load_images(img_path1, img_path2):
    # Charger les images en niveaux de gris
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    return img1, img2

def detect_and_describe(image):
    """ Détecte les points clés et extrait les descripteurs avec SIFT. """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Chemins des images
img_path1 = 'im1.jpeg'  # Chemin de l'image de gauche
img_path2 = 'im2.jpg'   # Chemin de l'image de droite

img1, img2 = load_images(img_path1, img_path2)
keypoints1, descriptors1 = detect_and_describe(img1)
keypoints2, descriptors2 = detect_and_describe(img2)
