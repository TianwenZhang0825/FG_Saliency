import os
import cv2

path = 'images_dir_path'

for i in os.listdir(path):
    image=cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE)
    
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    
    image = image * saliencyMap
    
    cv2.imwrite(os.path.join(path, i), image)