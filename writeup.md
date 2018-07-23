## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `camera_calibration` located in "./main.py#166"

![image-20180722204541320](https://ws1.sinaimg.cn/large/006tKfTcly1ftixe44itwj31ka0qiq93.jpg) 

The param of `camera_calibration` is `image_dir` and `params`

1st `image_dir` : directory which save chessboard images

2nd `params`: global params that save some parmeters for this function.

---

Here is the detail of this function implemention:

I start by preparing `object_points`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. 

Thus, `objp` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then I used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![image-20180722212643334](https://ws2.sinaimg.cn/large/006tKfTcly1ftiykrfdnrj30xk0d6ai0.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

1. Read image via function `read_image`:

![image-20180722212808063](https://ws4.sinaimg.cn/large/006tKfTcly1ftiym86gokj30z00budhy.jpg)

2. display this image by call function `show_image`:

   ![image-20180722212842662](https://ws2.sinaimg.cn/large/006tKfTcly1ftiymue24nj31k80p80y8.jpg)

3. Hers is the result:

![image-20180722205125048](https://ws2.sinaimg.cn/large/006tKfTcly1ftixk1zgq3j30vc0k4tqe.jpg)

4. And then, call `correct_distortion` to distortion image:

![image-20180722213003786](https://ws4.sinaimg.cn/large/006tKfTcly1ftiyo8wlh2j316w0budic.jpg)

5. Here is the distortion corrected image: 

![image-20180722205142288](https://ws2.sinaimg.cn/large/006tKfTcly1ftixkby35nj30vk0j84fu.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #178 in `main.py`):

![image-20180722214254975](https://ws4.sinaimg.cn/large/006tKfTcly1ftiz1mhr2zj31dk0qiq9p.jpg)

 Here's an example of my output for this step.  

![image-20180722214134061](https://ws3.sinaimg.cn/large/006tKfTcly1ftiz07p6elj30vg0k4afy.jpg)

The green color is comes from `sobel`, the blue color is comes from S channel in `HLS color space` 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform function is shown below:

![image-20180722223651044](https://ws1.sinaimg.cn/large/006tKfTcly1ftj0lqoizdj315k07odhr.jpg)

The `M` and `M_inv` matrix is calculate by `calc_perspective_transform`:

![image-20180722223746523](https://ws1.sinaimg.cn/large/006tKfTcly1ftj0mowi2vj311y1bidqi.jpg)

Use `cv2.getPerspectiveTransform()` to calculate a transform matrix. I also calculate a inverse  matrix call `M_inv` which is helpful for transform the image back from bird view. 

I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[weight, height - 10],  	# bottom right
                  [0, height - 10],  		# bottom left
                  [546, 460],  				# top left
                  [732, 460]])  			# top right
dst = np.float32([[weight, height],  		# bottom right
                  [0, height],  			# bottom left
                  [0, 0],  					# top left
                  [weight, 0]])  			# top right
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 1280, 710 |  1280, 720  |
|  0, 710   |   0, 720    |
| 546, 460  |    0, 0     |
| 732, 460  |   1280, 0   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![image-20180722225221254](https://ws2.sinaimg.cn/large/006tKfTcly1ftj11vc2z2j30vk0ck474.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use slide window to indentify lane line, here is the slide window code:

```python

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

    y_meters_per_pix = params['y_meters_per_pix']  # meters per pixel in y dimension
    x_meters_per_pix = params['x_meters_per_pix']  # meters per pixel in x dimension

    # Generate x and y values for plotting
    plot_y = np.linspace(0, image_height - 1, image_height)

    # plot read and blue lane mark
    out_img[non_zero_y[left_lane_indices], non_zero_x[left_lane_indices]] = [255, 0, 0]  # red
    out_img[non_zero_y[right_lane_indices], non_zero_x[right_lane_indices]] = [0, 0, 255]  # blue

    # Fit a second order polynomial to each (meters)
    left_fit_meters = np.polyfit(left_y * y_meters_per_pix, left_x * x_meters_per_pix, 2)
    right_fit_meters = np.polyfit(right_y * y_meters_per_pix, right_x * x_meters_per_pix, 2)

    # fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    logging.debug("left_fit: {}".format(left_fit_meters))
    logging.debug("right_fit: {}".format(right_fit_meters))

    return out_img, plot_y, left_fit_meters, right_fit_meters, left_fit, right_fit

```

* First I convert image from rgb color to gray

* And use `histogram = np.sum(image[image_height // 2:, :], axis=0)` to count a histgoram map like :

  > ![image-20180722225530204](https://ws3.sinaimg.cn/large/006tKfTcly1ftj1553hxdj30xa0patbx.jpg)

* And split this dirgram to two part(left and right). Find the peak of the dirgram:

  > ![image-20180722225651992](https://ws3.sinaimg.cn/large/006tKfTcly1ftj16k48lyj30yk05ita6.jpg)

* And then use slide window technique to count all `non_zero pixels`:

> ![image-20180722230014812](https://ws1.sinaimg.cn/large/006tKfTcly1ftj1a2p4h3j31es1aogxa.jpg)

* And use `np.polyfit` to find a 2nd polynomial:

> ![image-20180722230113894](https://ws2.sinaimg.cn/large/006tKfTcly1ftj1b3u6i5j30ps03e3z4.jpg)

* Final is looks like:

> ![image-20180722225927878](https://ws3.sinaimg.cn/large/006tKfTcly1ftj199j9r0j30vu0kogo9.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #383# in my code in `main.py`

* First read param `y_meters_per_pix` and `x_meters_per_pix` 
* And use this two params to get meters in real world

![image-20180722230350766](https://ws1.sinaimg.cn/large/006tKfTcly1ftj1dtinemj319s06mwgj.jpg)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #453# in my code in `main.py` in the function `draw_shadow()` and function `draw_data()`.  Here is an example of my result on a test image:

![image-20180722230812028](https://ws1.sinaimg.cn/large/006tKfTcly1ftj1id6uawj30vm0k01g5.jpg)

`draw_shadow()` function:

![image-20180722230905865](https://ws3.sinaimg.cn/large/006tKfTcly1ftj1ja9cp9j31dm0wy7cq.jpg)

`draw_data()` function:

![image-20180722230932995](https://ws1.sinaimg.cn/large/006tKfTcly1ftj1jrjmflj314y0j0jvz.jpg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://v.youku.com/v_show/id_XMzczOTI5MjIxMg==.html?spm=a2h3j.8428770.3416059.1)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



## False positive

![image-20180722233902235](https://ws1.sinaimg.cn/large/006tKfTcly1ftj2eg07lpj30vu0kstcb.jpg)

In this pipeline, I use `histogram` to find the start point of the lane-line from the perspective transofromed image.(shown above) 

There maybe some problem there. Because in some case there maybe more than one peak in the left or right(shown at the below image), Even some fo the roas side shadow is coresponse the peak. This will cause some false positive. 

![image-20180722234411676](https://ws1.sinaimg.cn/large/006tKfTcly1ftj2jthq48j313a0ucn3w.jpg)

To Fix this, Maybe I can cut some useless image out from source image. Or use some smooth technique to merge two peak to one. 

## False positive part 2

In some frame the green area is not cover all lane line:

![](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/8029/1532299372/Screenshot_from_2018-07-22_23-39-23.png)

So I implemete a function `is_parallel` to evaluate the left and right line is parallel. here is the code:

![image-20180723152400745](https://ws4.sinaimg.cn/large/006tNc79ly1ftjtppzljjj30m20co76l.jpg)

* Line 533: begin to iter this image in y direction, start from `0` end at `image height` each step in `100` pixels
* Line 534~535 : calc `left_x` and `right_x` 
* Line 537~538 : calculate current lane width.  And add all width to a array
* Line 547: use  `std` function in `numpy` to calculate `standard deviation`. If the lane width in each point are same, the standard deviation will very small. Otherwise this value will be very big. 
* And Also add some debug code from plot some information that can make sure my code is OK, here is the debug out:

> ![image-20180723153052525](https://ws1.sinaimg.cn/large/006tNc79ly1ftjtwtim22j30fu09e3yx.jpg)
>
> At the left line, I plot some `●` symble. At the right line, I plot some `★` symble.

## Make curv value correct

By check the document provide by us government, I know the first left turn curv is about `1km` but in my video, this value is near 1.5~2.0km. I try to find out why, but still have no idea. 

Here is my work:

* I know the `dot line dot line dot` pattern is 30 meters long by check : <http://www.dot.ca.gov/trafficops/camutcd/docs/TMChapter6.pdf>
* And then, I check the `perspective transformed` image in my project and find some corresponding image: 

> ![image-20180722235837296](https://ws4.sinaimg.cn/large/006tKfTcly1ftj2yt7banj30uu0kg7ex.jpg)

* After check this I think the value that I use to make `perspective transform` is OK

  * Here is the perspective transform param that I use:

  > ```python
  > # width is 1280, height is 720
  > src = [
  >     [weight, height - 10],  # bottom right
  >     [0, height - 10],  		# bottom left
  >     [546, 460],  			# top left
  >     [732, 460]				# top right
  > ]
  >
  > dst = [
  >     [weight, height],  		# bottom right
  >     [0, height],  			# bottom left
  >     [0, 0],  				# top left
  >     [weight, 0] 			# top right
  > ]
  > ```

* But the problem is still there… 