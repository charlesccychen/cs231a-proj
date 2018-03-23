import cv2
import numpy as np
import matplotlib.pyplot as plt

# input_1, input_2 = 'quad-1.jpg', 'quad-2.jpg'
# input_1, input_2 = 'cake-1.jpg', 'cake-2.jpg'
input_1, input_2 = 'data/statue/images/B22.jpg', 'data/statue/images/B23.jpg'

cake1 = cv2.imread(input_1)
cake2 = cv2.imread(input_2)
if 'statue' not in input_1:
	cake1 = cv2.resize(cake1, (0,0), fx=0.2, fy=0.2)
	cake2 = cv2.resize(cake2, (0,0), fx=0.2, fy=0.2)
cake1_gray = cv2.cvtColor(cake1, cv2.COLOR_BGR2GRAY)
cake2_gray = cv2.cvtColor(cake2, cv2.COLOR_BGR2GRAY)

# def show_rgb_img(img):
#   plt.imshow(cv2.cvtColor(img, cv2.CV_32S))
#   plt.show()

# show_rgb_img(cake1)

def sift(gray_img):
  sift = cv2.xfeatures2d.SIFT_create()
  kp, desc = sift.detectAndCompute(gray_img, None)
  return kp, desc

# def sift(gray_img):  # actually orb
#   sift = cv2.ORB_create()
#   kp, desc = sift.detectAndCompute(gray_img, None)
#   return kp, desc

# plt.subplot(1, 2, 1)
cake1_kp, cake1_desc = sift(cake1_gray)
# plt.imshow(cv2.drawKeypoints(cake1_gray, cake1_kp, cake1))
# plt.subplot(1, 2, 2)
cake2_kp, cake2_desc = sift(cake2_gray)
# plt.imshow(cv2.drawKeypoints(cake2_gray, cake2_kp, cake2))
#plt.show()

N_MATCHES = 120

# create a BFMatcher object which will match up the SIFT features
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(cake1_desc, cake2_desc)

# Sort the matches in the order of their distance.
sorted_matches = sorted(matches, key = lambda x:x.distance)

match_img = cv2.drawMatches(
    cake1, cake1_kp,
    cake2, cake2_kp,
    sorted_matches[:N_MATCHES], cake2.copy(), flags=0)
# plt.figure(figsize=(12,6))
# plt.imshow(match_img);
# plt.show()

# Filter out matches using ransac
from fundamental_matrix_estimation import *
num_best_matches = 0
best_matches = []
for _ in range(30):
	sample = np.random.choice(len(matches), size=8, replace=True)
	cake1_pts = []
	cake2_pts = []
	for i in sample:
		cake1_pt = np.concatenate([cake1_kp[matches[i].queryIdx].pt, [1]], axis=0)
		cake2_pt = np.concatenate([cake2_kp[matches[i].trainIdx].pt, [1]], axis=0)
		cake1_pts.append(cake1_pt)
		cake2_pts.append(cake2_pt)
	cake1_pts = np.array(cake1_pts)
	cake2_pts = np.array(cake2_pts)
	F = normalized_eight_point_alg(cake1_pts, cake2_pts)
	# Evaluate against matches.
	num_matches = 0
	current_matches = []
	for match in matches:
		cake1_pt = np.concatenate([cake1_kp[match.queryIdx].pt, [1]], axis=0)
		cake2_pt = np.concatenate([cake2_kp[match.trainIdx].pt, [1]], axis=0)
		error = cake2_pt.dot(F).dot(cake1_pt)
		if np.abs(error) < 0.3:
			num_matches += 1
			current_matches.append(match)
	print 'num_matches', num_matches
	if num_matches > num_best_matches:
		num_best_matches = num_matches
		best_matches = current_matches


sorted_matches = sorted(best_matches, key = lambda x:x.distance)
best_matches = sorted_matches[:N_MATCHES]

match_img = cv2.drawMatches(
    cake1, cake1_kp,
    cake2, cake2_kp,
    best_matches, cake2.copy(), flags=0)
plt.figure(figsize=(12,6))
plt.imshow(match_img);
plt.show()


sparse_structure_l = []
for match in best_matches:
	cake1_pt = cake1_kp[match.queryIdx].pt
	cake2_pt = cake2_kp[match.trainIdx].pt
	sparse_structure_l.append([cake1_pt[0], cake1_pt[1], cake2_pt[0], cake2_pt[1]])
sparse_structure = np.array(sparse_structure_l).T
sparse_structure = np.array([[sparse_structure]])
sparse_structure.dump('sparse_matches.npy')

def visualize(cake1, cake2, structure):
	plt.imshow(cake1)
	plt.scatter(x=structure[:,0], y=structure[:,1], marker='x')
	plt.show()
	plt.imshow(cake2)
	plt.scatter(x=structure[:,2], y=structure[:,3], marker='x')
	plt.show()

visualize(cake1, cake2, np.array(sparse_structure_l))

# Determine safe matchable region.
def compute_safematch(img_gray):
	H, W = img_gray.shape
	disparities = np.zeros((H, W, 4))
	directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
	print 'img_gray[:,1:', img_gray[:,1:]
	print 'X', np.concatenate([img_gray[:,1:], np.zeros((H, 1))], axis=1)
	disparities[:, :, 0] = np.abs(img_gray - np.concatenate([img_gray[:,1:], np.zeros((H, 1))], axis=1))
	disparities[:, :, 1] = np.abs(img_gray - np.concatenate([np.zeros((H, 1)), img_gray[:,0:W-1]], axis=1))
	disparities[:, :, 2] = np.abs(img_gray - np.concatenate([img_gray[1:,:], np.zeros((1, W))], axis=0))
	disparities[:, :, 3] = np.abs(img_gray - np.concatenate([np.zeros((1, W)), img_gray[0:H-1,:]], axis=0))
	safematch = np.max(disparities, axis=2) > 7
	# print safematch
	plt.imshow(safematch)
	plt.show()
	return safematch


cake1_safematch = compute_safematch(cake1_gray)
cake2_safematch = compute_safematch(cake2_gray)

# Create ZNCC descriptors
half_window_size = 2
def zncc(img):
	import collections
	zncc_d = collections.defaultdict(lambda: np.zeros((1+2*half_window_size, 1+2*half_window_size)))
	h, w = img.shape
	for i in range(h):
		for j in range(w):
			if i < half_window_size or i + half_window_size >= h:
				zncc_d[(i, j)] = np.zeros((1+2*half_window_size, 1+2*half_window_size))
				continue
			if j < half_window_size or j + half_window_size >= w:
				zncc_d[(i, j)] = np.zeros((1+2*half_window_size, 1+2*half_window_size))
				continue
			patch = img[i-half_window_size:i+half_window_size+1,j-half_window_size:j+half_window_size+1]
			patch = patch.astype(float)
			patch -= np.mean(patch)
			patch /= np.linalg.norm(patch)
			zncc_d[(i, j)] = patch
	return zncc_d
zncc_1 = zncc(cake1_gray)
zncc_2 = zncc(cake2_gray)

# print zncc_1

# Create dense correspondences.
from Queue import PriorityQueue
matchable = True
q = PriorityQueue()
seen1 = set()
seen2 = set()
for c1x, c1y, c2x, c2y in sparse_structure_l:
	c1x, c1y = int(c1x), int(c1y)
	c2x, c2y = int(c2x), int(c2y)
	if not((c1x, c1y) in zncc_1 and (c2x, c2y) in zncc_2):
		continue
	zncc_distance = np.sum(zncc_1[(c1x, c1y)] * zncc_2[c2x, c2y])
	q.put((-zncc_distance-1, (c1x, c1y, c2x, c2y)))

num_matches = 0
matches = []
while num_matches < 20000 and not q.empty():
	(neg_dist, (c1x, c1y, c2x, c2y)) = q.get()
	dist = -neg_dist
	if dist < 0.8:
		continue
	if not cake1_safematch[c1x, c1y] or not cake2_safematch[c2x, c2y]:
		continue
	print 'num_matches@!', num_matches, (neg_dist, (c1x, c1y, c2x, c2y))
	if (c1x, c1y) in seen1:
		continue
	if (c2x, c2y) in seen2:
		continue
	seen1.add((c1x, c1y))
	seen2.add((c2x, c2y))
	matches.append([float(c1x), float(c1y), float(c2x), float(c2y)])
	num_matches += 1
	local_matches = []
	local_seen = set()
	if dist >= 0.8:
		for nx1 in range(c1x-half_window_size, c1x+half_window_size+1):
			for ny1 in range(c1y-half_window_size, c1y+half_window_size+1):
				for nx2 in range(c2x-half_window_size, c2x+half_window_size+1):
					for ny2 in range(c2y-half_window_size, c2y+half_window_size+1):
						zncc_distance = np.sum(zncc_1[(nx1, ny1)] * zncc_2[nx2, ny2])
						q.put((-zncc_distance, (nx1, ny1, nx2, ny2)))



print matches
print 'done'
visualize(cake1, cake2, np.array(matches))

dense_structure = np.array(matches).T
dense_structure = np.array([[dense_structure]])
dense_structure.dump('dense_matches.npy')


# dense_structure.dump('dense_matches.npy')

fundamental_matrices = [[F]]
fundamental_matrices = np.array(fundamental_matrices)
fundamental_matrices.dump('fundamental_matrices.npy')


