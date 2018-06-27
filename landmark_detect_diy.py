import face_alignment
import numpy as np
from skimage import io


"""
FUNCTION: Get all the 68 points landmark detect of a input image
EDITOR: Hanchun Shen
DATE: 2018-04-02
INPUT:
	img_path: the path of input img
RETURN:
	result: np.shape[n, 68, 2], n implies the total faces number
"""
def landmark_preds(img_path):
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
	input = io.imread(img_path)
	if input.size <= 1:
		print "Empty Img_path: {}".format(img_path)
		return None
	preds = fa.get_landmarks(input)
	result = np.array(preds)
	if result is None:
		print "landmark detect failed!\n"
		return None
	else:
		print "landmark extract successfully!\n"
		return result


landmarks = landmark_preds('./test/assets/aflw-test.jpg')

