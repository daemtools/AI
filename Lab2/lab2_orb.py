import cv2
import numpy as np
import argparse

def template_matching(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Прямой поиск изображения на изображении
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Получаем координаты углов
    h, w = template_gray.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Рисуем рамку на обнаруженной области
    img_result = img.copy()
    cv2.rectangle(img_result, top_left, bottom_right, (255, 0, 0), 2)
    return img_result

def feature_matching(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Используем ORB для поиска ключевых точек и дескрипторов
    orb = cv2.ORB_create()
    keypoints_img, descriptors_img = orb.detectAndCompute(img_gray, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(template_gray, None)

    # BFMatcher для сопоставления дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_template, descriptors_img)
    matches = sorted(matches, key=lambda x: x.distance)

    img_result = img.copy()

    if len(matches) > 4:
        # Используем гомографию
        src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template_gray.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        img_result = cv2.polylines(img_result, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        # Для прямоугольника
        match_points = np.float32([keypoints_img[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(match_points)

        # Рисуем прямоугольник вокруг обнаруженных точек
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    else:
        print("Not enough matches are found - {}/{}".format(len(matches), 5))

    img_matches = cv2.drawMatches(template, keypoints_template, img, keypoints_img, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_result, img_matches

def resize_if_needed(image, template):
    img_height, img_width = image.shape[:2]
    temp_height, temp_width = template.shape[:2]

    if temp_height > img_height or temp_width > img_width:
        scale_factor = min(img_height / temp_height, img_width / temp_width)
        template = cv2.resize(template, (int(temp_width * scale_factor), int(temp_height * scale_factor)))

    return template

def main():
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Template and feature matching with OpenCV.')
    parser.add_argument('template', help='Path to the template image')
    parser.add_argument('image', help='Path to the input image')
    args = parser.parse_args()

    # Загружаем изображения эталона и входного изображения
    template = cv2.imread(args.template)
    img = cv2.imread(args.image)

    # Проверяем успешность загрузки изображений
    if template is None or img is None:
        print("Error loading images.")
        return

    # Уменьшите размер шаблона, если это необходимо
    template = resize_if_needed(img, template)

    # Прямое сопоставление шаблона
    result_template_matching = template_matching(img, template)
    cv2.imshow('Template Matching', result_template_matching)

    # Сопоставление с использованием ключевых точек
    result_feature_matching, img_matches = feature_matching(img, template)
    cv2.imshow('Feature Matching Result', result_feature_matching)
    cv2.imshow('Feature Matches', img_matches)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
