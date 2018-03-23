import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    A = []
    # print 'points1', points1
    assert points1.shape[0] == points2.shape[0]
    n = points1.shape[0]
    for i in range(n):
        p = points1[i]
        p_p = points2[i]
        u, v, _ = p
        u_p, v_p, _ = p_p
        A.append([u * u_p, v * u_p, u_p, u * v_p, v * v_p, v_p, u, v, 1])
    A = np.array(A)
    u, s, v = np.linalg.svd(A)
    assert v.shape[0] == 9
    f = v[8]
    F_hat = np.array(
        [[f[0], f[1], f[2]],
         [f[3], f[4], f[5]],
         [f[6], f[7], f[8]]])
    U, S, V = np.linalg.svd(F_hat)
    S_1 = S
    S_1[2] = 0
    F = U.dot(np.diag(S_1)).dot(V)
    return F




'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    centroid1 = np.mean(points1, axis=0)
    points_centered_1 = points1 - centroid1 + np.array([0, 0, 1])
    points_centered_2d_1 = points_centered_1[:,0:2]
    mean_dist_1 = np.mean(np.linalg.norm(points_centered_2d_1, axis=1))
    lambda_1 = 2.0 / mean_dist_1
    T = np.array(
        [[lambda_1, 0, -lambda_1 * centroid1[0]],
         [0, lambda_1, -lambda_1 * centroid1[1]],
         [0, 0, 1]])
    centroid2 = np.mean(points2, axis=0)
    points_centered_2 = points2 - centroid2 + np.array([0, 0, 1])
    points_centered_2d_2 = points_centered_2[:,0:2]
    mean_dist_2 = np.mean(np.linalg.norm(points_centered_2d_2, axis=1))
    lambda_2 = 2.0 / mean_dist_2
    T_p = np.array(
        [[lambda_2, 0, -lambda_2 * centroid2[0]],
         [0, lambda_2, -lambda_2 * centroid2[1]],
         [0, 0, 1]])
    F_q = lls_eight_point_alg(points1.dot(T.T), points2.dot(T_p.T))
    F = T_p.T.dot(F_q).dot(T)
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.plot(points1.T[0], points1.T[1], 'ro')
    lines1 = points2.dot(F)
    for i in range(lines1.shape[0]):
        xs = []
        ys = []
        A, B, C = lines1[i]
        for x in [0, 512]:
            y = (-C-A*x)/B
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys)
    plt.axis((0, 512, 512, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.plot(points2.T[0], points2.T[1], 'ro')
    lines2 = points1.dot(F.T)
    for i in range(lines2.shape[0]):
        xs = []
        ys = []
        A, B, C = lines2[i]
        for x in [0, 512]:
            y = (-C-A*x)/B
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys)
    plt.axis((0, 512, 512, 0))
    plt.show()

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    distances = []
    lines1 = points2.dot(F)
    assert points1.shape[0] == lines1.shape[0]
    for i in range(points1.shape[0]):
        A, B, C = lines1[i]
        x = float(points1[i][0])
        y = float(points1[i][1])
        distance = np.abs(A * x + B * y + C) / np.sqrt(A*A + B*B)
        distances.append(distance)
    return np.average(distances)

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized
        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
