import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import sys
import argparse

from lshash.lshash_2_py3 import LSHash
from skimage.util import random_noise
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from object_detection_YOLO import YoloObjectDetection

import os
import time
import glob
import cv2


#video_dir =  '/home/faisal/Desktop/Video/Walking_1.mp4'

# objectDetetcion = YoloObjectDetection()
# objectDetetcion.url1 = video_dir
# objectDetetcion.cap = cv2.VideoCapture(objectDetetcion.url1)
# objectDetetcion.init_tf_session()

class hash_search():

    def __init__(self,f_range, h_length,type,h_function,n_hashes_per_table,n_of_NN,DSF,TVD):

        #self.my_files_1 = sorted(glob.glob('./Cropped_Image_2/Cropped_Image/*.jpg'),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.my_files_1 = sorted(glob.glob('./temp/person/*.jpg'),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        #self.my_files_1 = sorted(glob.glob('./Cropped_Image_2/Cropped_Image/*.jpg'),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        #self.model = ResNet50(weights='imagenet', pooling=max, include_top=False)
        self.k = 0
        self.my_feature = []  # All feature is to stored here

        self.range = 1000
        self.hash_length = h_length
        self.hash_type = type
        self.hash_function_type = h_function
        self.hash_functions_per_table = n_hashes_per_table
        self.number_of_nearest_neighbours = n_of_NN
        self.downsampling_factor = DSF

        self.video_dir = TVD
        self.objectDetetcion = YoloObjectDetection()
        self.objectDetetcion.url1 = self.video_dir
        self.objectDetetcion.cap = cv2.VideoCapture(self.objectDetetcion.url1)
        self.objectDetetcion.init_tf_session()


    def init_lsh(self):

        self.lsh = LSHash(hash_size=self.hash_length, input_dim=self.range,
                          num_hashtables=1, num_hash_per_tables=self.hash_functions_per_table,
                          hash_function= self.hash_function_type,hash_type=self.hash_type,
                          no_of_nearest_neighbour=self.number_of_nearest_neighbours,
                          storage_config=None,matrices_filename=None,
                          overwrite=False)


        print(" \n LSH object instantiated \n ")


    def avg_downsample(self,feature):

        f= feature.reshape(-1, self.downsampling_factor).mean(axis=1)
        return f


    def get_vgg_feature(self, im):

        im = cv2.resize(im, (224, 224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        features = np.array(features)
        features = np.ravel(features)

        return features


    def add_noise_into_img(self,img):

        img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        return img

    def add_salt_pepper_noise(self,img):

        # Add salt-and-pepper noise to the image.
        noise_img = random_noise(img, mode='s&p', amount=0.001)

        # The above function returns a floating-point image
        # on the range [0, 1], thus we changed it to 'uint8'
        # and from [0,255]

        noise_img = np.array(255 * noise_img, dtype='uint8')

        return noise_img

    def plot_gallery(self,query_img, results):

        h = 100
        w = 100
        n_row = 3
        n_col = 5

        """Helper function to plot a gallery of portraits"""

        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)

            if i == 2 or i > 4:

                plt.imshow(cv2.cvtColor(cv2.resize(query_img, (h, w)),cv2.COLOR_BGR2RGB))

                if i == 2:
                    plt.title("QUERY", size=12)

                else:
                    plt.imshow(cv2.cvtColor(cv2.resize(cv2.imread(results[i - 5][0]), (h, w)),cv2.COLOR_BGR2RGB))
                    plt.title(str(i - 5), size=12)

            plt.xticks(())
            plt.yticks(())

        # plt.plot()
        # plt.pause(2)
        #plt.clf()

    def preprocess_all_features(self):

        for i , f in enumerate(self.my_feature):
            self.my_feature[i]= self.avg_downsample(f)


        self.range = self.my_feature[i].shape[0]

    def preprocess_current_feature(self, features):
        features = self.avg_downsample(features)
        return features



    def read_new_features(self,imgRange,var):

        for index in range(imgRange):  # Here the number 100

            im = cv2.imread(self.my_files_1[index])
            im = cv2.resize(im, (224, 224))
            #im = self.histogram_equalization(img_in= im)
            cv2.imshow("Image", im)
            cv2.waitKey(1)

            features = self.get_vgg_feature(im)
            features = self.avg_downsample(features)


            self.my_feature.append(features)

            print("feature reading index ", index)
            #return self.my_feature

        self.range = features.shape[0]

        if var == '1':
            np.save('/media/faisal/347b6790-47d5-4813-a0ec-b1ea4b766112/Data_Storage/OFD/Resnet-50/features_resnet-50', self.my_feature)

        elif var == '2':
            np.save('/media/faisal/347b6790-47d5-4813-a0ec-b1ea4b766112/Data_Storage/OFD/VGG-16/features_vgg-16', self.my_feature)

        elif var == '3':
            np.save('/media/faisal/347b6790-47d5-4813-a0ec-b1ea4b766112/Data_Storage/OFD/MobileNet-SSD/features_mobilenet-ssd', self.my_feature)


        cv2.destroyAllWindows()


    def test_blur_img(self,imgRange):

        plt.figure(figsize=(1.8 * 3, 2.4 * 5))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

        while True:

            imgs, frame = self.objectDetetcion.get_cropped_image()

            cv2.imshow(" video frame ", frame)
            #cv2.waitKey(1)

            k = cv2.waitKey(1)

            if k == 27:
                break

            elif cv2.waitKey(33) & 0xFF == ord('q'):


                for im in imgs:

                    print(" \n Press Enter ")
                    im = cv2.resize(im, (224, 224))

                    #im = self.add_noise_into_img(im)
                    im = self.add_salt_pepper_noise(im)

                    features = self.get_vgg_feature(im)
                    features = self.preprocess_current_feature(features)

                    print(features.shape)

                    result = self.query_image(img_feature=features)


                    if cv2.waitKey(0) :

                        #cv2.imshow(" query_image ", im)
                        self.plot_gallery(query_img = im,results=result)

                        plt.plot()
                        plt.pause(0.00001)
                        #plt.clf()

    def modelSelect(self,var):

        if(var=='1'):
            self.model= ResNet50(weights='imagenet', pooling=max, include_top=False)
        elif(var=='2'):
            self.model = VGG19(weights='imagenet', pooling=max, include_top=False)
        elif(var=='3'):
            self.model = MobileNet(weights='imagenet', pooling=max, include_top=False)

    def load_features_from_DB(self,var):

        print(" Loading features ")

        if var =='1':
            self.my_feature = np.load('/media/faisal/347b6790-47d5-4813-a0ec-b1ea4b766112/Data_Storage/OFD/Resnet-50/features_resnet-50.npy')
        elif var == '2':
            self.my_feature = np.load ('/media/faisal/347b6790-47d5-4813-a0ec-b1ea4b766112/Data_Storage/OFD/VGG-16/features_vgg-16.npy')
        elif var == '3':
            self.my_feature = np.load('/media/faisal/347b6790-47d5-4813-a0ec-b1ea4b766112/Data_Storage/OFD/MobileNet-SSD/features_mobilenet-ssd.npy')

        self.range = self.my_feature.shape[1]


        #print (self.model.summary)

    def indexing_feature(self,feature,additional_data, indx):

        self.lsh.index(feature[0:self.range], additional_data)

        print("indexing ::", indx)
        # print("indexing ::", additional_data)
        # print("indexing ::", feature)


    def query_image(self,img_feature):

        s = time.time()
        query_result = self.lsh.query(img_feature[0:self.range], num_results=10, distance_func='normalised_block_euclidean')
        e = time.time()

        print(" \n query_time ", e-s, '\n')
        return query_result


    def hashing_object_images(self,imgRange):

        for index, f in enumerate(self.my_feature):

            if index >=imgRange:
                break

            self.indexing_feature(feature=f,additional_data=(self.my_files_1[index],np.var(f)) ,indx = index)


def main(args):

    start = time.time()

    HL = args.hash_length
    HT = args.type
    HF = args.function
    N_HPT = args.n_of_HPT
    N_of_NN = args.n_of_NN
    feature_DSF = args.DSF
    video_path = args.TVD

    ########### Initiate search Object ######################################

    #range, h_length, type, h_function, n_hashes_per_table, n_of_NN, DSF

    svc = hash_search(f_range= 1,h_length=HL,type=HT,h_function=HF,
                      n_hashes_per_table=N_HPT,n_of_NN=N_of_NN,DSF=feature_DSF,TVD= video_path)

    ################## Number of Image to be read from Image ##############

    imgRange = args.range

    ############### Get Feature From Images  ###############
    print(args.RNF)

    if args.RNF:

        print("\n Select one of pretrained model for feature extraction \n")
        print(" 1:Resnet50\n 2:VGG19\n 3:MobilenetSSD\n")
        var = input()

        ############ Select Pretrained Object Detection Model #############

        svc.modelSelect(var)

        ############### Get Feature From Images  ###############

        svc.read_new_features(imgRange,var)

    else :

        print("\n Select one of saved features \n")

        print(" 1:Resnet50\n 2:VGG19\n 3:MobilenetSSD\n")
        var = input()

        svc.load_features_from_DB(var)
        svc.modelSelect(var)
        #svc.preprocess_all_features()



    end = time.time()
    print('\n\n time spend: ', (end - start) / 60, ' minutes \n\n')

    #################### Initiate LSH Hashing ####################

    #svc.range = np.array(svc.my_feature).shape[1]

    svc.init_lsh()

    ############## Start object hashing ################

    svc.hashing_object_images(imgRange)

    ############## Build Nearest Neighbour #############

    svc.lsh.build_NN(svc.lsh.hash_keys_array, 500)

    ################## Test Blur Images #################

    svc.test_blur_img(imgRange)

    # print(np.array(svc.my_feature).shape)
    cv2.destroyAllWindows()


def parse_arguments(argv):


    parser = argparse.ArgumentParser()


    parser.add_argument('--range', type=int,
                        help='Number of images to be read from directory')

    parser.add_argument('--hash_length', type=int,
                        help='length of hash key to be generated from features')

    parser.add_argument('--type', type=str, choices=['bin', 'discrete'],
                        help='type of hash keys "Binary and Discrete" hash are avalable')

    parser.add_argument('--function', type=str, choices=['pca', 'random'],
                        help='type of hash function')

    parser.add_argument('--n_of_HPT', type=int,
                        help='Number of hash function per table')

    parser.add_argument('--n_of_NN', type=int,
                        help='Number of nearest neighbour for NN search')

    parser.add_argument('--RNF', type=bool,
                        help='Read New Features (RNF) from object image directory',default=False)

    parser.add_argument('--DSF', type=int,
                        help=' Downsampling Factor ', default=2)

    parser.add_argument('--TVD', type=str,
                        help=' Test Video Path ', default= '/home/faisal/Desktop/Video/Walking_1.mp4')

    parser.add_argument('--OID', type=str,
                        help=' Object Image Directory ', default =None)

    parser.add_argument('--OFD', type=str,
                        help=' Object Features Directory ', default=None)




    return parser.parse_args(argv)

# python cluster_new1.py --range 200 --hash_length 48 --type discrete --function pca --n_of_HPT 5 --n_of_NN 20  --DSF 16 --RNF True

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


