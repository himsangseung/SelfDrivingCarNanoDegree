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

        # Testing
        self.category_index = {1:"Green", 10:"Red"}

        # tf detection session config 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # get tensor by name
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
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
        class_name = 'Red'
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                #rospy.loginfo("category_idx : %d", classes[i])
                class_idx = classes[i]
                if class_idx != 10 or class_idx != 1:
                    class_idx = 10 # default - testing

                class_name = self.category_index[class_idx]
        # Testing Phase on detection logic - not accurate
        if class_name == 'Red':
            self.current_light = TrafficLight.RED
        else:
            self.current_light = TrafficLight.GREEN

        rospy.loginfo("curr_light: %d",self.current_light)
        return self.current_light
        # return TrafficLight.UNKNOWN
