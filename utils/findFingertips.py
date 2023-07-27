import argparse
import cv2
import imutils
import math
import os
import numpy as np
from imutils import perspective
from imutils import contours
from tqdm import tqdm

# partially from https://github.com/ML-Dev-Alex/Fingertip-Detection-With-OpenCV/blob/master/findFingertips.py
def show_images(images, prefix="prefix", image_name="image"):
    """
    Displays images on screen and saves them on the hard-drive.
    :param images: List of cv2 images to display and save.
    :param prefix: Image prefix, generally a description of what kind of transformation was applied to the image.
    :param image_name: Name of the current image being displayed, a new folder will be created for the image name supplied.
    :return: Nothing.
    """

    # Creates output folders if they do not exist.
    # if not os.path.isdir(f"output"):
    #     os.mkdir(f"output")
    # if not os.path.isdir(f"output/{image_name}"):
    #     os.mkdir(f"output/{image_name}")

    # For each image supplied
    for i, img in enumerate(images):
        # Reduce the size of the visualization if the image is too big.
        cv2.namedWindow(f"{prefix}_{i}")
        if img.shape[0] > 1000:
            tmp_img = cv2.resize(img, (int(img.shape[1] / 6), int(img.shape[0] / 6)))
        else:
            tmp_img = img
        # Organize the display windows on the screen for better visualization.
        cv2.moveWindow(f"{prefix}_{i}", i * int(tmp_img.shape[1] + 50) + 200, 0)
        cv2.imshow(f"{prefix}_{i}", tmp_img)

    # Display every image at the same time, if you want to visualize images one by one, move the following
    # couple of lines to the inside of the for loop above.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def midpoint(ptA, ptB):
    """
    Simple support function that finds the arithmetic mean between two points.
    :param ptA: First point.
    :param ptB: Second point.
    :return: Midpoint.
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def crop_rect(img, rect, cut=True):
    """
    Crops a rectangle on an image and returns the rotated result.
    :param img: Input image to be cropped.
    :param rect: List of 4 points of the rectangle to crop the image.
    :param cut: Boolean to determine whether to actually cut the image before returning, or not.
    :return: Either a cropped image, or the full image with the rectangle drawn into it depending on the cut variable.
    """
    box = cv2.boxPoints(rect)  # to convert the Box2D structure given by cv2.minAreaRect() to 4 corner points
    box = np.int0(box)  # convert the 4 corners to the integer type
    if not cut:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 10)

    # take width and height of the rectangle
    W = rect[1][0]
    H = rect[1][1]

    # find max x and min x, max y and min y
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2] - 90

    if angle < -45:
        angle += 90
        rotated = True

    mult = 1.0
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))

    if not cut:
        cv2.circle(img, center, 10, (0, 255, 0), -1)
        size = (img.shape[0] + int(math.ceil(W)), img.shape[1] + int(math.ceil(H)))
        center = (int(size[0] / 2), int(size[1] / 2))

    # returns the transformation matrix M which will be used for rotating the image of "angle" degrees
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    return cropped


def auto_canny(image, sigma=0.33):
    """
    Automatically finds the best params to detect edges on a gray image based on the median values of its pixels.
    :param image: Grayscale image.
    :param sigma: Hyper-parameter to determine how open or closed the threshold should be.
    (The lower the sigma, the higher the range).
    :return: Image containing the edges of the original image.
    """
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)

    return edged


if __name__ == "__main__":

    # accept and read 3 command line arguments: mode, input path and output path
    parser = argparse.ArgumentParser(
        description="Script to cut fingertips from segmented finger photos and resize them to a common size",
        add_help=True,
    )
    parser.add_argument("--mode", help='mode of operation: "crop" or "resize"', default="crop", type=str)
    parser.add_argument("-i", "--input", help="input path", required=True, type=str)
    parser.add_argument("-o", "--output", help="output path", required=True, type=str)
    args = parser.parse_args()

    if args.mode == "resize":  # resize images to common size

        tot_width = 0
        tot_height = 0
        max_width = 0
        max_height = 0
        min_width = 100000
        min_height = 100000
        tot_aspect_ratio = 0

        max_width_name = ""
        max_height_name = ""
        min_width_name = ""
        min_height_name = ""

        counter = 0
        inputdir = args.input  # input path

        for subdir, dirs, files in os.walk(inputdir):
            # subdir is the current folder, dirs are the subfolders, files are the files in the folder
            for file in tqdm(files):
                name = os.path.join(subdir, file)
                img = cv2.imread(name)
                height, width, channels = img.shape
                tot_width += width
                tot_height += height
                tot_aspect_ratio += width / height
                counter += 1
                if width > max_width:
                    max_width = width
                    max_width_name = name
                if height > max_height:
                    max_height = height
                    max_height_name = name
                if width < min_width:
                    min_width = width
                    min_width_name = name
                if height < min_height:
                    min_height = height
                    min_height_name = name

        avg_width = tot_width // counter
        avg_height = tot_height // counter
        avg_aspect_ratio = tot_aspect_ratio / counter
        print(f"Max width: {max_width}, name: {max_width_name}")
        print(f"Average width: {avg_width}")
        print(f"Min width: {min_width}, name: {min_width_name}")
        print()
        print(f"Max height: {max_height}, name: {max_height_name}")
        print(f"Average height: {avg_height}")
        print(f"Min height: {min_height}, name: {min_height_name}")
        print()
        print(f"Average aspect ratio: {avg_aspect_ratio}")

        outputdir = args.output  # output path
        for subdir, dirs, files in os.walk(inputdir):
            # subdir is the current folder, dirs are the subfolders, files are the files in the folder
            for file in tqdm(files):

                if not os.path.isdir(outputdir + "/" + subdir.split("/", 1)[1]):
                    os.makedirs(outputdir + "/" + subdir.split("/", 1)[1])

                name = os.path.join(subdir, file)
                img = cv2.imread(name)

                resized = cv2.resize(img, (avg_width, avg_height), interpolation=cv2.INTER_AREA)

                cv2.imwrite(outputdir + "/" + subdir.split("/", 1)[1] + "/" + file, resized)

        quit()

    # change the folder string to the correct images folder location.
    folder = args.input  # input path
    output_path = args.output  # output path

    # create output path if it doesn't exist
    if not os.path.isdir(f"{output_path}"):
        os.makedirs(f"{output_path}")
    number_of_images = len(os.listdir(folder))

    for i, image_name in tqdm(enumerate(os.listdir(folder))):
        biggestArea = 0
        biggestContour = None
        orig_img = cv2.imread(f"{folder}/{image_name}")
        orig_img = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        done = False
        alpha = 2  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        height_factor = 3  # 3 usually better, use 2 to cut more
        while not done:
            img = orig_img.copy()  # reset image

            # enhances contrast of the image
            contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

            # blur to keep only the biggest edges
            blur = cv2.bilateralFilter(contrast, 7, 150, 150)

            # threshold to select only the brightest pixels of the image
            thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_OTSU)[1]

            # canny edge detection algorithm
            canny = auto_canny(thresh)

            kernel = np.array(
                (
                    [
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                    ]
                ),
                dtype=np.uint8,
            )
            eroded = cv2.dilate(canny, kernel, iterations=1)

            kernel = np.array(
                (
                    [
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                    ]
                ),
                dtype=np.uint8,
            )
            edged = cv2.dilate(eroded, kernel, iterations=1)

            kernel = np.array(
                (
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                dtype=np.uint8,
            )

            edged = cv2.dilate(edged, kernel, iterations=2)

            # Create lines on the image to encase fingers inside a box in order to generate a proper contour.
            cv2.line(
                edged,
                (0, edged.shape[0] - int(edged.shape[0] / height_factor)),
                (edged.shape[1], edged.shape[0] - int(edged.shape[0] / height_factor)),
                (255, 0, 255),
                10,
            )
            # show_images([edged], "EDGED", image_name)
            cv2.rectangle(
                edged,
                (0, edged.shape[0] - int(edged.shape[0] / height_factor)),
                (edged.shape[1], edged.shape[0]),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                img,
                (0, edged.shape[0] - int(edged.shape[0] / height_factor)),
                (edged.shape[1], edged.shape[0]),
                (0, 0, 0),
                -1,
            )  # draws a black filled rectangle on the original image

            kernel = np.ones((9, 9), np.uint8)
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

            cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # cnts is a tuple, cnts[0] shape: [856,1,2]

            # sort the contours from left-to-right
            (cnts, _) = contours.sort_contours(cnts)
            if len(cnts) > 1:
                cnts = (cnts[0],)

            # Create a copy of the edged image in order to be able to perform more transformations in the image.
            orig = edged.copy()
            biggestContours = []
            for j, c in enumerate(cnts):
                # if the contour is not sufficiently large, ignore it
                area = cv2.contourArea(c)
                height_rect = cv2.minAreaRect(c)[1][0]

                # if using height_factor = 2
                # if height_factor <= 2 and height_rect > 1100:
                #     height_factor -= 0.1
                #     continue
                if height_factor >= 3 and height_rect < 1150:
                    height_factor += 0.1
                    continue
                if area < ((orig.shape[0] * orig.shape[1]) / 13):  # originally / 10
                    height_factor += 0.1
                    continue
                else:
                    biggestContours.append(c)

                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # Order the points in the contour such that they appear in
                # top-left, top-right, bottom-right, and bottom-left order,
                # then draw the outline of the rotated bounding box
                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 10)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                # Unpack the ordered bounding box,
                (tl, tr, br, bl) = box

                # Then compute the midpoint between the top-left and top-right coordinates,
                (tltrX, tltrY) = midpoint(tl, tr)
                # Followed by the midpoint between bottom-left and bottom-right coordinates
                (blbrX, blbrY) = midpoint(bl, br)

                # Compute the midpoint between the top-left and top-right points,
                (tlblX, tlblY) = midpoint(tl, bl)
                # Followed by the midpoint between the top-right and bottom-right
                (trbrX, trbrY) = midpoint(tr, br)

                # Draw the midpoints on the image
                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # Draw lines between the midpoints
                cv2.line(
                    orig,
                    (int(tltrX), int(tltrY)),
                    (int(blbrX), int(blbrY)),
                    (255, 0, 255),
                    2,
                )
                cv2.line(
                    orig,
                    (int(tlblX), int(tlblY)),
                    (int(trbrX), int(trbrY)),
                    (255, 0, 255),
                    2,
                )

                # Check if current contour has the biggest area in all of the contours, and save it if so.
                if area >= biggestArea:
                    biggestArea = area
                    biggestContour = c

            # If we found a contour, draw the biggest one on the image and display it.
            if len(biggestContours) > 0:
                contours_image = cv2.drawContours(img.copy(), biggestContours, -1, (255, 0, 0), 10)

            # Otherwise, reduce the alpha (lower the contrast on the image), and try again, until it is done,
            # or until we reach the original contrast of the image.
            if biggestArea != 0 or alpha <= 1 or height_factor >= 20:
                done = True

        # If we still haven't found any contours on the image, print the image name and move on to the next one.
        if len(biggestContours) <= 0:
            print(f"Could not find contours on the image {image_name}.")
            continue

        # Select the bounding box of the biggest contour on the image (hopefully the hand).
        box = cv2.minAreaRect(biggestContour)  # returns a Box2D: (center(x, y), (width, height), angle of rotation)

        # Crop only the fingertip out of the original image
        finger = crop_rect(img, box)

        # crop finger image to left and right, removing black pixels
        rot_finger = cv2.rotate(finger, cv2.ROTATE_90_CLOCKWISE)
        left_idx = -1
        right_idx = -1
        for idx, column in enumerate(rot_finger):
            if np.any(column) and left_idx == -1:
                left_idx = idx - 1
            if np.all(column == 0) and left_idx != -1 and right_idx == -1:
                right_idx = idx
        # crop with left and right index found
        finger = finger[:, left_idx : right_idx + 1, :]

        cv2.imwrite(f"{output_path}/{image_name}", finger)
