import cv2
import numpy as np
import os

# Définition des chemins pour les sorties
output_directory = "output_results"
os.makedirs(output_directory, exist_ok=True)

def detect_and_describe(image):
    """ Détecte les points clés et extrait les descripteurs avec SIFT. """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def flann_matcher(descriptors1, descriptors2, trees, checks):
    """ Utilise FLANN pour matcher les descripteurs avec les paramètres donnés. """
    index_params = dict(algorithm=1, trees=trees)
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    return matches

def filter_matches_by_distance_and_direction(matches, keypoints1, keypoints2, max_distance=10, max_angle=10):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pt1 = keypoints1[m.queryIdx].pt
            pt2 = keypoints2[m.trainIdx].pt
            # Calculer la distance et la direction
            distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180 / np.pi
            # Filtrer selon la distance et la direction horizontale
            if distance <= max_distance and abs(angle) <= max_angle:
                good_matches.append(m)
    return good_matches

def main():
    # Chargement des images
    img1 = cv2.imread('im1.jpeg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)

    # Détecter les points clés et les descripteurs
    keypoints1, descriptors1 = detect_and_describe(img1)
    keypoints2, descriptors2 = detect_and_describe(img2)

    # Matcher les descripteurs
    matches = flann_matcher(descriptors1, descriptors2, trees=5, checks=50)
    # Filtrer les correspondances par distance et direction
    final_matches = filter_matches_by_distance_and_direction(matches, keypoints1, keypoints2)

    # Afficher ou sauvegarder les résultats
    print(f"Number of final good matches: {len(final_matches)}")
    result_text = f"Final good matches: {len(final_matches)}\n"
    with open(os.path.join(output_directory, "final_results.txt"), "w") as file:
        file.write(result_text)

if __name__ == "__main__":
    main()
