import cv2
import numpy as np

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

# convert from pixels to angles
# screen width (px) = 320
# screen height (px) = 240
# screen FOV x (deg) = 59.6
# screen FOV y (deg) = 49.7
def px_to_deg(cx, cy):
    tx = ((cx - 160.0) / 320.0) * 59.6
    ty = ((cy - 120.0) / 240.0) * 49.7
    return tx, -ty

def draw_point(image, x, y):
    cv2.circle(image, (int(x), int(y)), 5, (0,0,255), cv2.FILLED)

def draw_point_2(image, x, y):
    cv2.circle(image, (int(x), int(y)), 10, (0,0, 255), cv2.FILLED)

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(frame):
    try:
        # convert the input frame to the HSV color space
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        upper =  np.array([90, 200, 255], dtype = np.uint8)
        lower = np.array([75, 75, 0], dtype = np.uint8)
        img_threshold = cv2.inRange(img_hsv, lower, upper)
        
        cv2.imwrite("mask.jpg", img_threshold)

        # find contours in the new binary image
        contours, _ = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        points = []
        llpython = []
        largestContour = [[]]

        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 4: continue
            for coord in contours[i]:
                points.append(coord[0])

        x = []
        y = []
        for point in points:
            x.append(point[0])
            y.append(point[1])
        e = fit_ellipse(np.array(x), np.array(y))
        params = cart_to_pol(e)
        if params is None: return
        x, y = get_ellipse_pts(params)
        for i in range(len(x)):
            draw_point_2(frame, x[i], y[i])
        max = 9999
        for i in range(len(y)):
            if y[i] < max: max = y[i]

        draw_point(frame, int(params[0]), int(params[1] - params[3]))

        tx, ty = px_to_deg(params[0], abs(max))

        # initialize an array of values to send back to the robot
        llpython = [1, tx, ty, 0, 0]
        cv2.imwrite("hi.jpg", frame)
        
        # draw the bounding circle
                
        return np.array(largestContour), frame, llpython
    except:
        return np.array([[]]), frame, [0, 0, 0, 0, 0]
    
if __name__ == "__main__":
    # run through camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # add gausian blur
        img = cv2.GaussianBlur(img, (101, 101), 0)
        
        
        # run the pipeline
        _, processed_img, _ = runPipeline(img)
        
        # display the output
        cv2.imshow("output", processed_img)
        
        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
