import os
import rospy
import cv2
import numpy as np
from styx_msgs.msg import TrafficLight
import yaml
import tensorflow as tf


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # light_config_str = rospy.get_param("/traffic_light_config")
        # self.config = yaml.safe_load(light_config_str)

        # Graph - tf model loading
        #self.is_site = self.config['is_site']

        # added
        # pass
        model_file = "frozen_inference_graph.pb" 
        self.current_light = TrafficLight.UNKNOWN

        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, "model/{}".format(model_file))
        # rospy.logwarn("model_path={}".format(model_path))

        # load tf model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #self.category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}

        # Testing
        self.category_index = {1:"Green", 10:"Red"}

        # tf detection session init
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')


        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # return TrafficLight.UNKNOWN

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)

        # Actual detection
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
            self.detection_classes, self.num_detections],             feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        min_score_thresh = .5
        count = 0
        count1 = 0
        class_name = 'Red'
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                count1 += 1
                rospy.loginfo("category_idx : %d", classes[i])
                #class_name = self.category_index[classes[i]]['name']
                class_name = self.category_index[classes[i]]
                # Traffic light thing
                if class_name == 'Red':
                    count += 1

        # if count < count1 - count:
        #    self.current_light = TrafficLight.GREEN
        #else:
        #     self.current_light = TrafficLight.RED
        
        if class_name == 'Red':
            self.current_light = TrafficLight.RED
        else:
            self.current_light = TrafficLight.GREEN

        rospy.loginfo("curr_light: %d",self.current_light)
        # return self.current_light
        return TrafficLight.UNKNOWN
