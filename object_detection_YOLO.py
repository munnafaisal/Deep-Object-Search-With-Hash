import tensorflow as tf

k= int(tf.__version__.split('.')[0])
if k >=2:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

import tensornets as nets
import cv2
import numpy as np
import time
import argparse
import sys
from create_folder import createFolder

tf.disable_v2_behavior()

class YoloObjectDetection():

    def __init__(self):

        self.url1 = []
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.model = nets.YOLOv3COCO(self.inputs, nets.Darknet19)
        self.cap = []
        self.query_obj_type = []
        # model = nets.YOLOv2(inputs, nets.Darknet19)

        # frame=cv2.imread("D://pyworks//yolo//truck.jpg",1)
        self.count = 0
        self.classes = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'bike', '5': 'bus', '7': 'truck', '8': 'chair'}
        self.list_of_classes = [0, 1, 2, 3, 5, 7, 8]

    def init_tf_session(self):

            with tf.Session().as_default() as self.sess:

                self.sess.run(self.model.pretrained())

    def get_cropped_image(self):


        ret, frame = self.cap.read()
        img = cv2.resize(frame, (416, 416))
        copy_img = img.copy()
        imge = np.array(img).reshape(-1, 416, 416, 3)

        start_time = time.time()
        preds = self.sess.run(self.model.preds, {self.inputs: self.model.preprocess(imge)})

        #print("--- %s seconds ---" % (time.time() - start_time))
        boxes = self.model.get_boxes(preds, imge.shape[1:3])

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #
        # cv2.resizeWindow('image', 700, 700)
        # print("--- %s seconds ---" % (time.time() - start_time))
        boxes1 = np.array(boxes)

        img_list = []
        box_list = []

        for j in self.list_of_classes:
            count = 0

            if str(j) in self.classes:
                lab = self.classes[str(j)]

                if lab == self.query_obj_type:

                    if len(boxes1) != 0:

                        for i in range(len(boxes1[j])):

                            box = boxes1[j][i]

                            if boxes1[j][i][4] >= 0.5:
                                count += 1


                                crop_img = copy_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                                # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                                # cv2.putText(img, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),
                                #             lineType=cv2.LINE_AA)

                                img_list.append(crop_img)
                                box_list.append(box)


        return box_list,img_list, img

    def crop_and_save(self):


        with tf.Session() as sess:

            sess.run(self.model.pretrained())
            # "D://pyworks//yolo//videoplayback.mp4"
            cap = cv2.VideoCapture(self.url1)

            while (cap.isOpened()):

                ret, frame = cap.read()
                img = cv2.resize(frame, (416, 416))
                copy_img = img.copy()
                imge = np.array(img).reshape(-1, 416, 416, 3)

                start_time = time.time()
                preds = sess.run(self.model.preds, {self.inputs: self.model.preprocess(imge)})

                print("--- %s seconds ---" % (time.time() - start_time))
                boxes = self.model.get_boxes(preds, imge.shape[1:3])
                
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)

                cv2.resizeWindow('image', 700, 700)
                # print("--- %s seconds ---" % (time.time() - start_time))
                boxes1 = np.array(boxes)

                for j in self.list_of_classes:
                    count = 0
                    if str(j) in self.classes:
                        lab = self.classes[str(j)]
                    if len(boxes1) != 0:

                        for i in range(len(boxes1[j])):

                            box = boxes1[j][i]

                            if boxes1[j][i][4] >= 0.5:

                                count += 1

                                crop_img = copy_img[int(box[1]):int(box[3] ), int(box[0]):int(box[2])]

                                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                                cv2.putText(img, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),
                                            lineType=cv2.LINE_AA)



                                cv2.imshow("cropped_image", crop_img)
                                cv2.waitKey(1)

                                dir = "temp/" + lab + "/"
                                createFolder(dir)
                                s1 = dir + '{}.jpg'.format(self.count)
                                self.count = self.count + 1
                                print("\n object_count :", self.count)
                                cv2.imwrite(s1, crop_img)

                    #print(lab, ": ", count)

                cv2.imshow("image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main(args):

    objectDetetcion = YoloObjectDetection()
    objectDetetcion.url1 = args.video_dir
    objectDetetcion.cap = cv2.VideoCapture(objectDetetcion.url1)

    objectDetetcion.crop_and_save()


def parse_arguments(argv):

    parser = argparse.ArgumentParser()


    parser.add_argument('--video_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

#objectDetetcion.init_tf_session()

# for i in range(1000):
#     print("Press Enter ")
#     imgs = objectDetetcion.get_cropped_image()
#
#     for index, img in enumerate(imgs) :
#         cv2.waitKey(0)
#
#         if index == 3:
#             break
#
#         cv2.imshow(str(index), img)
#
#     print(len(imgs))
