import cv2
from cv2 import HOGDescriptor
import numpy as np
import argparse

parser = argparse.ArgumentParser("HOG + SVM")
parser.add_argument("--svm", default='mnist.svm')
args = parser.parse_args()

x_train = np.load('x_test.npy')
y_train = np.load('y_test.npy')
# x_train = x_train.reshape(-1, 784).astype("float32") / 255
y_train = y_train.astype("int32")

hog = cv2.HOGDescriptor()

svm = cv2.ml.SVM_create()
print("Loading pretrained SVM")
svm.load('mnist.svm')
print("Finish")

# hog.setSVMDetector(svm)

def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    print(sv)

get_svm_detector(svm)

# vector< float > get_svm_detector( const Ptr< SVM >& svm )
# {
#     // get the support vectors
#     Mat sv = svm->getSupportVectors();
#     const int sv_total = sv.rows;
#     // get the decision function
#     Mat alpha, svidx;
#     double rho = svm->getDecisionFunction( 0, alpha, svidx );
#     CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
#     CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
#                (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
#     CV_Assert( sv.type() == CV_32F );
#     vector< float > hog_detector( sv.cols + 1 );
#     memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
#     hog_detector[sv.cols] = (float)-rho;
#     return hog_detector;
# }
