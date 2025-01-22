import math
import time
import cv2
import numpy as np

VISION_L_H = 67
VISION_L_S = 103
VISION_L_V = 51
VISION_U_H = 128
VISION_U_S = 255
VISION_U_V = 255
MIN_AREA = 1000
ECCENTRICITY = 0.7 #0.51
PERCENTAGE = 0.7 

##### CHECK FOR RESULTS IN testouput FOLDER!!

def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Assumes the ellipse is 'axis aligned' - modelled by setting the cross product term equal to 1

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """

    #Quadratic terms of the design matrix
    D1 = np.vstack([x**2, np.ones(len(x)), y**2]).T
    # Linear terms of the design matrix
    D2 = np.vstack([x, y, np.ones(len(x))]).T

    #Sections of the scatter matrix
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigenval, eigenvec = np.linalg.eig(M)
    con = 4 * eigenvec[0]* eigenvec[2] - eigenvec[1]**2
    ak = eigenvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the Cartesian coefficients, (a, b, c, d, e, f), to polar parameters, 
    x0, y0, ap, bp, e, phi, where 
    (x0, y0) is the ellipse centre; 
    (ap, bp) are the semi-major and semi-minor axes, respectively; 
    e is the eccentricity; and 
    phi is the rotation of the semi-major axis from the x-axis.

    """

    if(len(coeffs)==0): 
        return

    # assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    det = b**2 - a*c
    # if det > 0:
    #     raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
    #                      ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / det, (a*f - b*d) / det

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / det / (fac - a - c))
    bp = np.sqrt(num / det / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    # phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=50, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """
    if params is None: return

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def draw_point(image, x, y):
    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), 2)

def draw_point_2(image, x, y):
    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), 2)

def ellipse_detect(frame, img_threshold, contour):
    global MIN_AREA, ECCENTRICITY, PERCENTAGE
    if cv2.contourArea(contour) < 4: return 0.0, -360.0

    points = []
    for coord in contour:
        points.append(coord[0])

    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    e = fit_ellipse(np.array(x), np.array(y))
    params = cart_to_pol(e)
    
    if params is None: return 0.0, -360.0

    if (params[4] > ECCENTRICITY): # Eccentricity check
        return 0.0, -360.0
    
    image2 = np.zeros_like(img_threshold)
        
    image2 = cv2.ellipse(image2, (int(params[0]), int(params[1])), (int(params[2]), int(params[3])), 0, 
                            0, 360, (255, 255, 255) , -1)
    bitwise = cv2.bitwise_and(img_threshold, image2)
    number_pixels = cv2.countNonZero(bitwise)
    area_ellipse = math.pi * params[2] * params[3]
    percentage = number_pixels / area_ellipse
    print("PERCENTAGE ", percentage)

    if area_ellipse < MIN_AREA: # Area check
        #print("MIN_AREA", MIN_AREA.valueInt())
        return 0.0, -360.0

    if percentage < PERCENTAGE: # Algae color check
        return 0.0, -360.0

    x, y = get_ellipse_pts(params)
    for i in range(len(x)):
        draw_point_2(frame, x[i], y[i])
    max = 9999
    for i in range(len(y)):
        if y[i] < max: max = y[i]

    draw_point(frame, int(params[0]), int(params[1] - params[3]))

    tx, ty = params[0], abs(max)

    return tx, ty


def runPipeline(frame):
    global VISION_L_H, VISION_L_S, VISION_L_V, VISION_U_H, VISION_U_S, VISION_U_V, MIN_AREA, ECCENTRICITY, PERCENTAGE

    frame = cv2.GaussianBlur(frame, (51, 51), 0)
    try:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # upper =  np.array([90, 200, 255], dtype = np.uint8)
        # lower = np.array([75, 75, 0], dtype = np.uint8)

        lower =  np.array([VISION_L_H, VISION_L_S, VISION_L_V], dtype = np.uint8)
        upper = np.array([VISION_U_H, VISION_U_S, VISION_U_V], dtype = np.uint8)

        img_threshold = cv2.inRange(img_hsv, lower, upper)
        #img_threshold = cv2.GaussianBlur(img_threshold, (51, 51), 0)

        img_threshold[img_threshold > 3] = 255

        contours, _ = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        ellipses = []

        img_threshold_3c = cv2.cvtColor(img_threshold, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite("thresholds.jpg", img_threshold_3c)
        frame = cv2.addWeighted(frame, 0.6, img_threshold_3c, 0.4, 0) #adds the mask on top of the image

        for contour in contours:
            tx, ty = ellipse_detect(frame, img_threshold, contour)
            # if ty > 0.0: # Check that circle is in bottom half of frame
            #     ellipses.append([tx, ty])
            ellipses.append([tx, ty])

        
        return frame, ellipses, threshold
    except Exception as e:
        print(f"the exception is {type(e).__name__}")

        return frame, [], None

if __name__ == "__main__":

    for i in range (0, 1701): #1827
        try:
            img = cv2.imread("./ALGAE_IMAGES_AUGMENTED_RESIZED/algae_image_1_" + str(i) + ".jpg")
            height, width = img.shape[:2]
            t_start = time.time_ns()

            if (height < width):
                img = cv2.resize(img, (240, 180))
            elif (height > width):
                img = cv2.resize(img, (180, 240))

            processed_img, ellipses, threshold = runPipeline(img)

            print("FPS: ", 1e9 / (time.time_ns() - t_start))
            cv2.imwrite("./testoutput/output1_" + str(i) + ".jpg", processed_img)
            if threshold is not None:
                cv2.imwrite("./threshold/output1_" + str(i) + ".jpg", threshold)
                print("HERE")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except TypeError:
            print(str(i) + " failed")

    # cap = cv2.VideoCapture(0)
    # # create_sliders()

    # ctr = 0

    # while True:
    #     ret, img = cap.read()
    #     if not ret:
    #         break

    #     t_start = time.time_ns()
        
    #     img = cv2.GaussianBlur(img, (101, 101), 0)
        
    #     processed_img, ellipses = runPipeline(img)

    #     if ctr == 50:
    #         print("FPS: ", 1e9 / (time.time_ns() - t_start))
    #         ctr = 0
    #     ctr += 1

    #     cv2.imshow("output", processed_img)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # cap.release()
    cv2.destroyAllWindows()