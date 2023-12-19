import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import convolve
from scipy import fft
from skimage import color, restoration, io
import cv2

K = 190 # shutter speed

focal_length = 1386.6
im_shape = (1920, 1080)
starting_point = np.array([0, 0, 1])
calib = np.array([[focal_length, 0, 0],[0, focal_length, 0], [0, 0, 1]])
highpass = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
sobel_x = np.array([[1, 0, -1],
                   [2,  0, -2],
                   [1, 0, -1]]) 
sobel_y = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

def grad(img):
    xDiff = np.diff(img, axis=0, append=np.expand_dims(
        np.take(img, 0, axis=0), axis=0))
    yDiff = np.diff(img, axis=1, append=np.expand_dims(
        np.take(img, 0, axis=1), axis=1))
    return xDiff, yDiff

def compute_gaussian(x, sigma, mu):
    c = 1 / (sigma * np.sqrt(2*np.pi))
    return c * np.exp(-((x-mu)**2) / (2*(sigma**2)))

def readSensorData(fname):
    gyro, accel = [], []
    f = open(fname, "r")
    for i, x in enumerate(f):
        d_arr = x.split(",")
        float_arr = np.asarray(d_arr, dtype=float)
        if i % 2 == 0:
            accel.append(float_arr)
        else:
            gyro.append(float_arr)
    return np.array(gyro), np.array(accel)

def readSensorDataGyro(fname):
    gyro_d = []
    f = open(fname, "r")
    for x in f:
        d_arr = x.split(",")
        float_arr = np.asarray(d_arr, dtype=float)
        gyro_d.append(float_arr)
    gyro_d = np.array(gyro_d)
    w_x = gyro_d[:, 1]
    w_y = gyro_d[:, 2]
    w_z = gyro_d[:, 0]

    modified_gyro = np.zeros(gyro_d.shape)
    modified_gyro[:, 0] = w_x
    modified_gyro[:, 1] = w_y
    modified_gyro[:, 2] = w_z
    return modified_gyro

def getRotationData(gyro_data):
    gyro_data *= (np.pi/180)
    thetas = [np.array([0, 0, 0])]
    R_init = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Rs = [R_init]
    for i in range(1, K):
        theta_k = (gyro_data[i-1])*(1/1000) + thetas[i-1]
        R_i = Rotation.from_rotvec(theta_k).as_matrix()
        thetas.append(theta_k)
        Rs.append(R_i)
    return np.array(Rs), np.array(thetas)

def sensor_gaussian(sensor_data):
    x_pts = sensor_data[:, 0]
    y_pts = sensor_data[:, 1]

    x_mean = np.mean(x_pts)
    x_var = np.var(x_pts)
    y_mean = np.mean(y_pts)
    y_var = np.var(y_pts)

    return x_mean, x_var, y_mean, y_var
    
def gen_blur_pts(gyro_data):
    xaxis = np.arange(0, len(gyro_d))
    yaxis_1 = gyro_data[:, 0]
    yaxis_2 = gyro_data[:, 1]
    plt.plot(xaxis, yaxis_1, label=r"$\omega_x$")
    plt.plot(xaxis, yaxis_2, label=r"$\omega_y$")
    plt.legend(loc='best')
    plt.xlabel("Time (ms)")
    plt.ylabel("Angular velocity (DPS)")
    # plt.savefig("./res/movie/mangular_vel.jpg")
    plt.show()
    

    x_0 = starting_point[0]
    y_0 = starting_point[1]
    pts_k_x = []
    pts_k_y = []

    Rs, thetas = getRotationData(gyro_data)
    plt.plot(np.arange(0, len(thetas)), thetas[:, 0], label=r"$\theta_x$")
    plt.plot(np.arange(0, len(thetas)), thetas[:, 1], label=r"$\theta_y$")
    plt.legend(loc='best')
    plt.xlabel("Time (ms)")
    plt.ylabel("Angular position (radians)")
    # plt.savefig("./res/movie/mangular_pos.jpg")
    plt.show()
    
    for i in range(0, K):
        # tt = (calib @ (Rs[i] @ np.linalg.inv(calib))) @ starting_point
        # x_i = tt[0] / tt[2]
        # y_i = tt[1] / tt[2]
        theta_x, theta_y = thetas[i][0], thetas[i][1]
        x_i = x_0 + (focal_length*theta_y)
        y_i = y_0 + (focal_length*theta_x)
        pts_k_x.append(x_i)
        pts_k_y.append(y_i)
    return np.array(pts_k_x), np.array(pts_k_y)

def initial_psf(pts_k_x, pts_k_y, x_mean, x_var, y_mean, y_var):
    # 75 x 75 kernel
    # x_range = np.linspace(-3*3, 1*5, 75)
    # y_range = np.linspace(-3*3, 1*5, 75)
    # x_range = np.linspace(-4, 3, 75)
    # y_range = np.linspace(-2, 7, 75)
    x_range = np.linspace(-5, 1, 50)
    y_range = np.linspace(-1, 5, 50)
    psf = []
    for y in y_range[::-1]:
        curr_row = []
        for x in x_range[::-1]:
            curr_pt = 0
            for k in range(0, K):
                x_k = pts_k_x[k]
                y_k = pts_k_y[k]
                curr_pt += compute_gaussian(x - x_k, 3*np.sqrt(x_var), x_mean) * compute_gaussian(y-y_k, 3*np.sqrt(y_var), y_mean)
            curr_pt /= K
            curr_row.append(curr_pt)
        psf.append(curr_row)
    return np.array(psf)

def coherence_filter(img, sigma = 11, str_sigma = 11, blend = 0.5, iter_n = 4):
    h, w = img.shape[:2]

    for i in range(iter_n):
        gray = img
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]

        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img*(1.0 - blend) + img1*blend)
    print('done')
    return img

def strong_edge_prediction(img):
    tau = 0.2
    bilat = cv2.bilateralFilter(img.astype('float32'),9,75,75)
    bilat = (bilat*255).astype('uint8')
    sf = coherence_filter(bilat)
    dx, dy = grad(sf)
    grad_sf = np.array([dx, dy])
    heaviside_f = np.heaviside(np.linalg.norm(grad_sf) - tau*np.max(np.linalg.norm(grad_sf)), 0.5)
    strong_edges = grad_sf*heaviside_f
    return strong_edges

def estimate_psf(grad_se, grad_b):
    se_dx, se_dy = grad_se[0], grad_se[1]
    b_dx, b_dy = grad_b[0], grad_b[1]
    gamma = 80

    a1 = np.conj(fft.fft2(se_dx).T)@fft.fft2(b_dx)
    print(a1)
    a2 = np.conj(fft.fft2(se_dy).T)@fft.fft2(b_dy)
    K = np.real(fft.ifft2((a1 + a2) / (a1 + a2 + gamma)))
    maxElem = np.max(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            elem = K[i][j]
            if elem < (0.02*maxElem):
                K[i][j] = 0.0
    print(np.sum(K))
    K = K / np.sum(K)
    return K

def calculate_activity_map(blurred_img):
    pt = 1500
    res_map = np.zeros(blurred_img.shape)
    for i in range(blurred_img.shape[0]):
        for j in range(blurred_img.shape[1]):
            curr_var = 0.01
            if i > 1 and i < blurred_img.shape[0]-2 and j > 1 and j < blurred_img.shape[1]:
                curr_cluster = blurred_img[i-2:i+3, j-2:j+3].flatten()
                curr_var = np.var(curr_cluster)
            res_map.append(1/(pt*curr_var + 1))
    return res_map

def gain_map(img):
    alpha = 0.2
    l = 5
    sm = 0
    for i in range(l):
        rows, cols = map(int, img.shape)
        img = cv2.pyrDown(img, dstsize=(cols // 2, rows // 2))
        dx, dy = grad(img)
        sm += np.linalg.norm(np.array([dx, dy]))/300
    print(sm)
    return 1-alpha + alpha*sm

def richardson_lucy(image, psf, num_iter=10, clip=True, filter_epsilon=None):
    im_deconv = np.full(image.shape, 0.5)
    psf_mirror = np.flip(psf)

    # Small regularization parameter used to avoid 0 divisions
    eps = 1e-12

    for i in range(num_iter):
        print(i)
        conv = convolve(im_deconv, psf, mode='same') + eps
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        im_deconv *= convolve(relative_blur, psf_mirror, mode='same')
        if i - 1 < num_iter:
            im_deconv *= gain_map(im_deconv)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def iterate_psf(start_k, blurred_img):
    blurred_img = color.rgb2gray(blurred_img)
    num_iters = 1
    curr_k = start_k
    b_dx, b_dy = grad(blurred_img)
    for i in range(num_iters):
        print(i)
        curr_deconvolve = restoration.wiener(color.rgb2gray(blurred_img), curr_k, 1100)
        s_dx, s_dy = strong_edge_prediction(curr_deconvolve)
        curr_k = estimate_psf(np.array([s_dx, s_dy]), np.array([b_dx, b_dy]))
    return curr_k

def deconvolve_rl(img, psf):
    res = []
    for i in range(3):
        deconvolved_RL = richardson_lucy(img[:, :, i], psf, num_iter=30)
        res.append(deconvolved_RL)
    return np.dstack((res[0],res[1],res[2]))

def deconvolve_wiener(img, psf):
    res = []
    for i in range(3):
        deconvolved_w = restoration.wiener(img[:, :, i], psf, 1100)
        res.append(deconvolved_w)
    return np.dstack((res[0],res[1],res[2]))

img1_noisy = io.imread("./data/rockfish.JPG") / 255
gyro_d = readSensorDataGyro("./data/sensor_rockfish2.txt")
gaussian_gyro = readSensorDataGyro("./data/sensor_gaussian.txt")
x_mean, x_var, y_mean, y_var = sensor_gaussian(gaussian_gyro)
# plt.hist(gaussian_gyro[:, 0], bins=45, label="X data")
# plt.hist(gaussian_gyro[:, 1], bins=45, label="Y data")
# plt.xlabel("Angular velocity")
# plt.ylabel("Number of occurences")
# plt.legend(loc='best')
# plt.savefig("./res/gaussian/noise_histo.jpg")
# plt.show()


xks, yks = gen_blur_pts(gyro_d)
plt.plot(-1*xks, yks)
# # plt.axis([-14, 14, -2, 9])
plt.xlabel("X location (pixels)")
plt.ylabel("Y location (pixels)")
plt.savefig("./res/rockfish/rpoint_path.jpg")
plt.show()

psf0 = initial_psf(xks, yks, x_mean, x_var, y_mean, y_var)
plt.imshow(psf0, cmap='gray')
plt.show()
io.imsave("./res/rockfish/rpsf.jpg", psf0)
# final_k = iterate_psf(psf0, img1_noisy)
# curr_deconvolve = deconvolve_wiener(img1_noisy, psf0)
# tt = restoration.wiener(color.rgb2gray(img1_noisy), psf0, 1100)
# plt.imshow(tt, cmap='gray')
# plt.show()
deconvolved_RL = deconvolve_rl(img1_noisy, psf0)
plt.imshow(deconvolved_RL)
plt.show()
# io.imsave("./res/rockfish/im_rockfish.jpg", deconvolved_RL)
# xaxis = np.arange(0, len(gyro_2))
# yaxis_3 = gyro_2[:, 2]
# plt.plot(xaxis, yaxis1)
# plt.plot(xaxis, yaxis_2)
# plt.hist(yaxis1, 50)
# plt.hist(yaxis_2, 50)
# plt.show()

# plt.scatter(yaxis1, yaxis_2)
# plt.show()
# img = io.imread("./res/rockfish/im_rockfish.jpg") / 255
# img = color.rgb2gray(img)
# bilat = cv2.bilateralFilter(img.astype('float32'),9,75,75)

# # display both images in a 1x2 grid
# fig = plt.figure()
# fig.add_subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# fig.add_subplot(1, 2, 2)
# plt.imshow(bilat, cmap='gray')
# plt.show()
