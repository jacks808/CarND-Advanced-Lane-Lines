import glob
import logging
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s]%(asctime)s - %(message)s \t\t@[%(filename)s:%(lineno)d] ',
                    filemode='w')

params = {
    'mode': 'camera_calibration',
    'mode': 'correct_distortion_show_case',
    'mode': 'perspective_transform_show_case',
    'mode': 'test_curv',
    'mode': 'image',
    'mode': 'video',

    'debug_mode': True,

    'ret_pickle_path': 'ret.pickle',
    'mtx_pickle_path': 'mtx.pickle',
    'dist_pickle_path': 'dist.pickle',
    'rvecs_pickle_path': 'rvecs.pickle',
    'tvecs_pickle_path': 'tvecs.pickle',
    "num_x": 9,
    "num_y": 6,
    "vis_corners": True,

    "s_thresh": (170, 255),
    'sx_thresh': (20, 100),
    'soble_threshold_y': 1.3,

    'num_windows': 9,
    'window_margin': 100,  # Set the width of the windows +/- margin
    'min_pix': 50,  # Set minimum number of pixels found to recenter window

    # curvature
    'y_meters_per_pix': 30 / 720,  # meters per pixel in y dimension
    'x_meters_per_pix': 3.7 / 700,  # meters per pixel in x dimension

    # cache for perspective transform
    'M_cache': None,
    'M_inv_cache': None,
}


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


DIST = pickle_load(params['dist_pickle_path'])
MTX = pickle_load(params['mtx_pickle_path'])


def read_image(filepath, gray=False):
    """
    read image
    :param filepath:
    :param gray:
    :return:
    """
    if gray:
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def show_histogram(histogram, title="", debug=False):
    if not debug:
        return

    plt.plot(histogram)
    plt.title(title)
    plt.show()


def show_image(img, title="", cmap=None, left_fit_x=None, right_fit_x=None, plot_y=None,
               force_output=params['debug_mode']):
    """
    show image
    :param img: image data
    :param cmap: color map
    :param title: title of a image
    :param left_fit_x:
    :param right_fit_x:
    :param plot_y:
    :param force_output:
    :return:
    """
    if force_output:  # not debug mode, then not output image file
        logging.debug("show image")

        if left_fit_x is not None and right_fit_x is not None and plot_y is not None:
            plt.plot(left_fit_x, plot_y, color='yellow')
            plt.plot(right_fit_x, plot_y, color='yellow')

        plt.imshow(X=img, cmap=cmap)
        plt.title(title)
        plt.show()
        return


def find_corners(img, params):
    """
    find corners
    :param img:
    :param params: use num_x, num_y
    :return:  ret, corners
    """
    num_x = params['num_x']
    num_y = params['num_y']
    return cv2.findChessboardCorners(img, (num_x, num_y), None)


def show_corners(img, corners, params):
    """
    show corners
    :param img:
    :param corners:
    :param params:
    :return:
    """
    num_x = params['num_x']
    num_y = params['num_y']
    return cv2.drawChessboardCorners(img, (num_x, num_y), corners, True)


def camera_calibration(image_dir, params):
    objp = np.zeros((params['num_y'] * params['num_x'], 3), np.float32)
    objp[:, :2] = np.mgrid[0:params['num_x'], 0:params['num_y']].T.reshape(-1, 2)

    object_points = []
    image_points = []

    image_paths = glob.glob(image_dir + "/*.jpg")

    for image_path in image_paths:
        logging.info("calibration for image {}".format(image_path))

        gray_image = read_image(image_path)

        found, corners = find_corners(gray_image, params)
        if found:
            object_points.append(objp)
            image_points.append(corners)

            if params['vis_corners']:
                show_image(show_corners(gray_image, corners, params))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_image.shape[::-1][1:], None,
                                                       None)
    return ret, mtx, dist, rvecs, tvecs


def correct_distortion(image, mtx, dist):
    """
    correct distortion
    :param image: image
    :param mtx: mtx
    :param dist: dist
    :return:
    """
    logging.debug("correct distortion image with mtx: {}, dist: {}".format(mtx, dist))
    correct_image = cv2.undistort(image, mtx, dist, newCameraMatrix=mtx)
    return correct_image


def image_to_binary(image):
    _, result = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return result


def color_and_gradient(img, params):
    s_thresh = params['s_thresh']
    sx_thresh = params['sx_thresh']

    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


def calc_perspective_transform(image):
    # cache
    if params['M_cache'] is not None and params['M_inv_cache'] is not None:
        return params['M_cache'], params['M_inv_cache']

    height, weight = image.shape[:2]

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([[weight, height - 10],  # bottom right
                      [0, height - 10],  # bottom left
                      [546, 460],  # top left
                      [732, 460]])  # top right

    # For source points I'm grabbing the outer four detected corners
    # src = np.float32([[weight, height - 10],  # bottom right
    #                   [0, height - 10],  # bottom left
    #                   [546, 450],  # top left
    #                   [732, 450]])  # top right

    if params['debug_mode']:
        plt.imshow(image)

        plt.plot(weight, height - 10, 'o')
        plt.plot(0, height - 10, 'o')
        plt.plot(546, 460, 'o')
        plt.plot(732, 460, 'o')

        plt.show()

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = np.float32([[weight, height],  # bottom right
                      [0, height],  # bottom left
                      [0, 0],  # top left
                      [weight, 0]])  # top right

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    params['M_cache'] = M
    params['M_inv_cache'] = M_inv

    return M, M_inv


def perspective_transform(image, par_transformams):
    M, M_inv = calc_perspective_transform(image)

    height, weight = image.shape[:2]

    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(image, M, (weight, height), flags=cv2.INTER_LINEAR)


def fit_lane(image, pamras):
    """
    plot and fit lane
    :param image: image
    :param pamras: params
    :return:
        out_img, out image after plot lane line
        plot_y,
        left_fit_meters,
        right_fit_meters,
        left_fit,
        right_fit
    """
    # convert iamge to gray scale
    image = convert_gray(image)

    # Take a histogram of the bottom half of the image
    image_height = image.shape[0]
    histogram = np.sum(image[image_height // 2:, :], axis=0)
    show_histogram(histogram, "Histogram of lanes", debug=params['debug_mode'])

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    mid_point = np.int(histogram.shape[0] // 2)
    left_x_base = np.argmax(histogram[:mid_point])
    right_x_base = np.argmax(histogram[mid_point:]) + mid_point

    # Choose the number of sliding windows
    num_windows = pamras['num_windows']

    # Set height of windows
    window_height = np.int(image_height // num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    non_zero = image.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])

    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Set the width of the windows +/- margin
    margin = params['window_margin']

    # Set minimum number of pixels found to recenter window
    min_pix = params['min_pix']

    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for window in range(num_windows):  # iter each window
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image_height - (window + 1) * window_height
        win_y_high = image_height - window * window_height

        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(img=out_img, pt1=(win_x_left_low, win_y_low), pt2=(win_x_left_high, win_y_high),
                      color=(0, 255, 0),  # green box
                      thickness=2)
        cv2.rectangle(img=out_img, pt1=(win_x_right_low, win_y_low), pt2=(win_x_right_high, win_y_high),
                      color=(0, 255, 0),  # green box
                      thickness=2)

        # Identify the non zero pixels in x and y within the window
        good_left_indices = (
                (non_zero_y >= win_y_low) &
                (non_zero_y < win_y_high) &
                (non_zero_x >= win_x_left_low) &
                (non_zero_x < win_x_left_high)
        ).nonzero()[0]

        good_right_indices = (
                (non_zero_y >= win_y_low) &
                (non_zero_y < win_y_high) &
                (non_zero_x >= win_x_right_low) &
                (non_zero_x < win_x_right_high)
        ).nonzero()[0]

        # Append these indices to the lists
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        # If you found > min_pix pixels, recenter next window on their mean position
        if len(good_left_indices) > min_pix:
            left_x_current = np.int(np.mean(non_zero_x[good_left_indices]))

        if len(good_right_indices) > min_pix:
            right_x_current = np.int(np.mean(non_zero_x[good_right_indices]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right line pixel positions
    left_x = non_zero_x[left_lane_indices]
    left_y = non_zero_y[left_lane_indices]
    right_x = non_zero_x[right_lane_indices]
    right_y = non_zero_y[right_lane_indices]

    # Generate x and y values for plotting
    plot_y = np.linspace(0, image_height - 1, image_height)

    # plot read and blue lane mark
    out_img[non_zero_y[left_lane_indices], non_zero_x[left_lane_indices]] = [255, 0, 0]  # red
    out_img[non_zero_y[right_lane_indices], non_zero_x[right_lane_indices]] = [0, 0, 255]  # blue

    y_meters_per_pix = params['y_meters_per_pix']  # meters per pixel in y dimension
    x_meters_per_pix = params['x_meters_per_pix']  # meters per pixel in x dimension

    # Fit a second order polynomial to each (meters)
    left_fit_meters = np.polyfit(left_y * y_meters_per_pix, left_x * x_meters_per_pix, 2)
    right_fit_meters = np.polyfit(right_y * y_meters_per_pix, right_x * x_meters_per_pix, 2)

    # fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    logging.debug("left_fit: {}".format(left_fit_meters))
    logging.debug("right_fit: {}".format(right_fit_meters))

    return out_img, plot_y, left_fit_meters, right_fit_meters, left_fit, right_fit


def convert_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def calc_center(image, params, left_fit, right_fit):
    x_meters_per_pix = params['x_meters_per_pix']  # meters per pixel in x dimension

    bottom_y = image.shape[0]  # height of this image is the bottom x value

    left_bottom_x = calc_polynomial(bottom_y, left_fit[0], left_fit[1], left_fit[2])
    right_bottom_x = calc_polynomial(bottom_y, right_fit[0], right_fit[1], right_fit[2])

    lane_center_position = (right_bottom_x - left_bottom_x) / 2 + left_bottom_x
    car_position = image.shape[1] / 2  # car position

    center_dist = (car_position - lane_center_position) * x_meters_per_pix
    return center_dist


def calc_polynomial(x, a, b, c):
    """
    y = ax^2 + bx + c
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """
    return a * x ** 2 + b * x + c


def calc_curv(plot_y, line_fit_params):
    y_eval = np.max(plot_y)
    return ((1 + (2 * line_fit_params[0] * y_eval + line_fit_params[1]) ** 2) ** 1.5) / np.absolute(
        2 * line_fit_params[0])


def draw_data(image, curv_rad, center_dist):
    new_img = np.copy(image)

    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (200, 70), font, 2.5, (255, 0, 0), 2, cv2.LINE_AA)

    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'

    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (200, 160), font, 2.5, (255, 0, 0), 2, cv2.LINE_AA)
    return new_img


def draw_shadow(origin_image, undist_image, plot_y, left_fit_x, right_fit_x, M_inv):
    """
    draw shadow on image
    :param origin_image: image
    :param undist_image: origin image for draw lane
    :param plot_y:
    :param left_fit_x:
    :param right_fit_x:
    :param M_inv:
    :return:
    """
    # Create an image to draw the lines on
    color_warp = np.zeros_like(origin_image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    show_image(color_warp, "Image after draw green area")

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (origin_image.shape[1], origin_image.shape[0]))
    show_image(newwarp, "Image after perspective transform back to origin image")

    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

    return result


def handle_image(image):
    """
    handle image
    :param image: image
    :return: handled image
    """
    show_image(image, title="Origin image")

    # 1. Distortion correction
    correct_image = correct_distortion(image, MTX, DIST)
    show_image(correct_image, title="Distortion corrected image")

    show_image(perspective_transform(correct_image, params), title="perspective transform image")

    # 2. color/ gradient threshold
    gradient_image = color_and_gradient(correct_image, params)
    show_image(gradient_image, title="Gradient Image")

    # 3. perspective transform
    transform_image = perspective(gradient_image, params)
    show_image(transform_image, title="Perspective transformed image")

    # 4. use slide window to plot line
    out_img, plot_y, left_fit_meters, right_fit_meters, left_fit, right_fit = fit_lane(transform_image, params)

    left_fit_x = calc_polynomial(plot_y, left_fit[0], left_fit[1], left_fit[2])
    right_fit_x = calc_polynomial(plot_y, right_fit[0], right_fit[1], right_fit[2])

    show_image(out_img, title="Image after plot lane",
               left_fit_x=left_fit_x, right_fit_x=right_fit_x, plot_y=plot_y)

    # 5. calc center distance of road
    center_dist = calc_center(image, params, left_fit, right_fit)

    # 6. calc curvature
    left_curv = calc_curv(plot_y, left_fit_meters)
    right_curv = calc_curv(plot_y, right_fit_meters)
    avg_curv = np.average([left_curv, right_curv])

    # 7. draw shadow
    M, M_inv = calc_perspective_transform(image)
    image_shadow = draw_shadow(image, image, plot_y, left_fit_x, right_fit_x, M_inv)
    show_image(image_shadow, title="Image after draw shadow")

    # 8. draw data
    image_add_curv_data = draw_data(image_shadow, avg_curv, center_dist)
    show_image(image_add_curv_data, title="Final image")

    return image_add_curv_data


def pickle_dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    mode = params['mode']

    logging.info("start at mode: {}".format(mode))
    if mode == 'camera_calibration':
        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = camera_calibration('./camera_cal', params)
        pickle_dump(ret, params['ret_pickle_path'])
        pickle_dump(mtx, params['mtx_pickle_path'])
        pickle_dump(dist, params['dist_pickle_path'])
        pickle_dump(rvecs, params['rvecs_pickle_path'])
        pickle_dump(tvecs, params['tvecs_pickle_path'])
        logging.info("camera calibration finish")

    elif mode == 'image':
        test_img_dir = 'test_images'
        for test_img in os.listdir(test_img_dir):
            image = read_image(os.path.join(test_img_dir, test_img))
            show_image(image, force_output=True)

            handled_image = handle_image(image)
            cv2.imwrite('output_images/{}'.format(test_img), handled_image)

            show_image(handled_image, force_output=True)
            # break

    elif mode == 'video':
        clip1 = VideoFileClip(filename="./project_video.mp4")#.subclip(0, 5)
        # clip1 = VideoFileClip(filename="./challenge_video.mp4").subclip(0, 5)
        white_clip = clip1.fl_image(handle_image)
        white_clip.write_videofile("./out_{}.mp4".format(time.time()), audio=False)
        pass
    elif mode == 'correct_distortion_show_case':
        # distortion_image
        origin_image = read_image("./test_images_2/straight_lines1.jpg")
        correct_distortion = correct_distortion(origin_image, MTX, DIST)

        plt.figure()

        plt.subplot(121)
        plt.imshow(origin_image)
        plt.title("origin image")

        plt.subplot(122)
        plt.imshow(correct_distortion)
        plt.title("distortion_image")

        plt.show()

    elif mode == 'perspective_transform_show_case':
        # perspective_transform
        origin_image = read_image("./test_images_2/straight_lines1.jpg")
        distortion = correct_distortion(origin_image, MTX, DIST)
        transformed = perspective_transform(distortion, params)

        plt.figure()

        plt.subplot(121)
        plt.imshow(origin_image)
        plt.title("origin image")

        plt.subplot(122)
        plt.imshow(transformed)
        plt.title("perspective transform image")

        plt.show()
    elif mode == 'test_curv':
        # perspective_transform
        origin_image = read_image("./test_images_2/straight_lines1.jpg")
        handle_image(origin_image)
        origin_image = read_image("./test_images/straight_lines2.jpg")
        handle_image(origin_image)

    else:
        logging.error("error")
