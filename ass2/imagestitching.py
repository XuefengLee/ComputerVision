from __future__ import print_function  #
import cv2
import numpy as np
import argparse
import os


def up_to_step_1(imgs):
	"""Complete pipeline up to step 3: Detecting features and descriptors"""

	imgs = detect(imgs)
	return imgs


def save_step_1(imgs, output_path='./output/step1'):
	"""Save the intermediate result from Step 1"""

	for filename,img,_,_ in imgs:

		cv2.imwrite(output_path + '/' + filename + '.jpg',img)


def up_to_step_2(imgs):
	"""Complete pipeline up to step 2: Calculate matching feature points"""
	
	data = detect(imgs)

	images = []
	length = len(data)
	for i in range(length):
		for j in range(i+1,length):
			images.append(matching(data[j], data[i]))

	return images


def save_step_2(imgs,  output_path="./output/step2"):
	"""Save the intermediate result from Step 2"""
	for img in imgs:
		cv2.imwrite(output_path+'/'+img[0]+'.jpg', img[1])

def up_to_step_3(imgs):

	data = detect(imgs)
	images = []
	length = len(data)

	for train in range(length):
		# index,img,kp,des = data[train]
		for query in range(train+1,length):
			_, img, good = matching(data[query],data[train])
			# images.append(img)
			H = ransac(good,data[query],data[train])

			Warp = linear_transformation(data[query][1], H)
			filename1 = data[query][0] + '_warp_' + data[train][0] + '_ref'

			inv = np.linalg.inv(H)
			Ref = linear_transformation(data[train][1], inv)
			filename2 = data[train][0] + '_warp_' + data[query][0] + '_ref'

			images.append(((filename1,Warp),(filename2,Ref)))

	return images

def save_step_3(img_pairs, output_path="./output/step3"):
	"""Save the intermediate result from Step 3"""

	for pair in img_pairs:
		cv2.imwrite(output_path+'/'+ pair[0][0]+'.jpg', pair[0][1])
		cv2.imwrite(output_path+'/'+ pair[1][0]+'.jpg', pair[1][1])


def up_to_step_4(imgs):
	"""Complete the pipeline and generate a panoramic image"""
	data = detect(imgs)
	length = len(data)

	info = []
	# for train in range(length - 1):
	# 	query = train + 1

	_, img, good = matching(data[0],data[1])
	H = ransac(good,data[0],data[1])
	info.append((H,img))
	_, img, good = matching(data[0],data[1])
	info.append((np.eye(3),data[1][1]))
	
	_, img, good = matching(data[2],data[1])
	H = ransac(good,data[2],data[1])
	info.append((H,img))
	constructImages(info)
	# return imgs[0]


def save_step_4(imgs, output_path="./output/step4"):
	"""Save the intermediate result from Step 4"""
	# ... your code here ...
	pass


def findH(src, dst):

	A = []
	for i in range(len(src)):
		x1, y1 = src[i][0], src[i][1]
		x2, y2 = dst[i][0], dst[i][1]
		A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
		A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

	U, S, Vh = np.linalg.svd(A)
	# L = Vh[-1,:] / Vh[-1,-1]
	L = Vh[-1,:]
	H = L.reshape(3, 3)

	return H

def detect(imgs):
	"""Complete pipeline up to step 3: Detecting features and descriptors"""

	data = []

	for filename,img in imgs:
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
		kp,des = sift.detectAndCompute(gray,None)
		img = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		data.append((filename,img,kp,des))

	return data

def matching(query, train):

	query_name, query_img, query_kp, query_des = query
	train_name, train_img, train_kp, train_des = train


	# compute Euclidean distance matrix
	dists = np.sqrt(((query_des[:, :, None] - train_des[:, :, None].T) ** 2).sum(1))

	matches = []

	# get k nearest match point
	for i in range(dists.shape[0]):
		index = dists[i].argsort()[:2]

		# throw non-symmetry nearest relation
		index_2 = dists[:,index[0]].argsort()[0]
		if index_2 != i:
			continue

		matches.append((cv2.DMatch(i, index[0], dists[i][index[0]]),cv2.DMatch(i, index[1], dists[i][index[1]])))



	good = []
	for a,b in matches:
		if a.distance < 0.7 * b.distance:
			good.append(a)

	img = cv2.drawMatches(query_img,query_kp,train_img,train_kp,good,None)

	filename = train_name+'_'+str(len(train_kp))+'_'+query_name+'_'+str(len(query_kp))+'_'+str(len(good))

	return (filename, img, good)

def ransac(good,query,train):

	src_pts = np.float32([ np.append(query[2][v.queryIdx].pt,1) for v in good ])
	dst_pts = np.float32([ train[2][v.trainIdx].pt for v in good ])

	max_num = 0
	Homo = None
	for _ in range(100):
		idx = np.random.randint(len(src_pts), size=4)
		H = findH(src_pts[idx,:], dst_pts[idx,:])

		out = np.matmul(H,src_pts.transpose()).transpose()

		inlier = np.sqrt((out[:,:2]/out[:,[-1]] - dst_pts)**2)
		inlier = np.where(inlier < 3, 1, 0)
		num = np.sum(inlier)

		# out = np.linalg.inv(H).dot(dst_pts)

		if num > max_num:
			max_num = num
			Homo = H

	return Homo


def linear_transformation(img, a):

	M, N, D = img.shape
	points = np.append(np.mgrid[0:N, 0:M].reshape((2, M*N)),[np.ones(M*N)],axis=0)

	affine_points = a.dot(points)
	affine_points /= affine_points[2, :]


	# find range of the new image
	minX = np.min(affine_points[0, :])
	minY = np.min(affine_points[1, :])
	maxX = np.max(affine_points[0, :])
	maxY = np.max(affine_points[1, :])

	rangeW = int(maxX - minX)
	rangeH = int(maxY - minY)

	indY, indX = np.indices((rangeH, rangeW))
	newPictureIndex = np.stack((indX.ravel(), indY.ravel(), np.ones(indY.size)))
	newPictureIndex[0, :] += minX
	newPictureIndex[1, :] += minY

	mapToOriginalImgPoints = np.linalg.inv(a).dot(newPictureIndex)

	mapToOriginalImgPoints /= mapToOriginalImgPoints[2, :]

	dst = np.ones([rangeH, rangeW, D])

	map_x, map_y = mapToOriginalImgPoints[:2].reshape((2,rangeH,rangeW)).round().astype(np.int)

	dst = np.zeros([rangeH, rangeW, D])

	for i in range(rangeH):
		for j in range(rangeW):
			indexX = map_x[i][j]
			indexY = map_y[i][j]
			if indexX > N-1 or indexY > M-1 or indexX < 0 or indexY < 0:
				continue
			value = img[indexY][indexX]
			dst[i][j] = value

	return dst

def findImageBorder(data):

    minXs = []
    minYs = []
    maxXs = []
    maxYs = []
    for H, img in data:
    	# print(img)
        height,width,depth = img.shape
        indY, indX = np.indices((height, width))
        rangeIndex = np.stack((indX.ravel(), indY.ravel(), np.ones(indY.size)))

        rangePoints = H.dot(rangeIndex)
        rangePoints /= rangePoints[2, :]
        minX = np.min(rangePoints[0, :])
        minY = np.min(rangePoints[1, :])
        maxX = np.max(rangePoints[0, :])
        maxY = np.max(rangePoints[1, :])
        minXs.append(minX)
        minYs.append(minY)
        maxXs.append(maxX)
        maxYs.append(maxY)

    return min(minXs), min(minYs), max(maxXs), max(maxYs)

def constructImages(data):
    minX, minY, maxX, maxY = findImageBorder(data)

    rangeW = int(maxX - minX)
    rangeH = int(maxY - minY)

    indY, indX = np.indices((rangeH, rangeW))
    newPictureIndex = np.stack((indX.ravel(), indY.ravel(), np.ones(indY.size)))
    newPictureIndex[0, :] += minX
    newPictureIndex[1, :] += minY

    dst = np.ones([rangeH, rangeW, 3])

    for H, img in data:
        
        currHeight, currWidth, depth = img.shape
        print(H)
        inverse = np.linalg.inv(H)
        mapToOriginalImgPoints = inverse.dot(newPictureIndex)

        mapToOriginalImgPoints /= mapToOriginalImgPoints[2, :]

        map_x = mapToOriginalImgPoints[0]
        map_y = mapToOriginalImgPoints[1]
        map_x = map_x.reshape(rangeH, rangeW).astype(np.int)
        map_y = map_y.reshape(rangeH, rangeW).astype(np.int)

        for i in range(rangeH):
            for j in range(rangeW):
                indexX = map_x[i][j]
                indexY = map_y[i][j]
                if indexX > currWidth - 1 or indexY > currHeight - 1 or indexX < 0 or indexY < 0:
                    continue
                value = img[indexY][indexX]
                dst[i][j] = value
    cv2.imwrite("step4.jpg",dst)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"step",
		help="compute image stitching pipeline up to this step",
		type=int
	)

	parser.add_argument(
		"input",
		help="a folder to read in the input images",
		type=str
	)

	parser.add_argument(
		"output",
		help="a folder to save the outputs",
		type=str
	)

	args = parser.parse_args()

	imgs = []
	filelist = os.listdir(args.input)
	filelist.sort()
	for filename in filelist:
		print(filename)
		img = cv2.imread(os.path.join(args.input, filename))
		imgs.append((os.path.splitext(filename)[0],img))

	if args.step == 1:
		print("Running step 1")
		modified_imgs = up_to_step_1(imgs)
		save_step_1(modified_imgs, args.output)
	elif args.step == 2:
		print("Running step 2")
		modified_imgs = up_to_step_2(imgs)
		save_step_2(modified_imgs, args.output)
	elif args.step == 3:
		print("Running step 3")
		img_pairs = up_to_step_3(imgs)
		save_step_3(img_pairs, args.output)
	elif args.step == 4:
		print("Running step 4")
		panoramic_img = up_to_step_4(imgs)
		save_step_4(panoramic_img, args.output)

