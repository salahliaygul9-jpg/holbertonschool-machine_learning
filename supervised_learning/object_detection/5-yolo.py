#!/usr/bin/env python3
"""
 Preprocess images
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


class Yolo:
    """
    Uses Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        process outputs
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            boxes.append(output[..., 0:4])
        box_confidences = \
            [self.sigmoid(output[..., 4, np.newaxis]) for output in outputs]
        box_class_probs = \
            [self.sigmoid(output[..., 5:]) for output in outputs]
        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape
            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)
            idxs_y = np.arange(grid_height)
            idxs_y = idxs_y.reshape(grid_height, 1, 1)
            cy = c + idxs_y
            idxs_x = np.arange(grid_width)
            idxs_x = idxs_x.reshape(1, grid_width, 1)
            cx = c + idxs_x
            tx = (box[..., 0])
            ty = (box[..., 1])
            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)
            bx = tx_n + cx
            by = ty_n + cy
            bx /= grid_width
            by /= grid_height
            tw = (box[..., 2])
            th = (box[..., 3])
            tw_t = np.exp(tw)
            th_t = np.exp(th)
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = pw * tw_t
            bh = ph * th_t
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            box[..., 0] = x1 * image_width
            box[..., 1] = y1 * image_height
            box[..., 2] = x2 * image_width
            box[..., 3] = y2 * image_height
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Flter boxes
        """
        scores = []

        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            scores.append(box_conf * box_class_prob)

        max_scores = [score.max(axis=3) for score in scores]
        max_scores = [score.reshape(-1) for score in max_scores]
        box_scores = np.concatenate(max_scores)
        index_to_delete = np.where(box_scores < self.class_t)
        box_scores = np.delete(box_scores, index_to_delete)

        box_classes_list = [box.argmax(axis=3) for box in scores]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, index_to_delete)

        filtered_boxes_list = [box.reshape(-1, 4) for box in boxes]
        filtered_boxes_box = np.concatenate(filtered_boxes_list, axis=0)
        filtered_boxes = np.delete(filtered_boxes_box, index_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou(box1, box2):
        """intersection over union
        (x1, y1, x2, y2)"""
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        intersection_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)
        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non max suppression
        """
        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predicted_box_classes = np.array([box_classes[i] for i in index])
        predicted_box_scores = np.array([box_scores[i] for i in index])

        _, class_counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        accumulated_count = 0

        for class_count in class_counts:
            while i < accumulated_count + class_count:
                j = i + 1
                while j < accumulated_count + class_count:
                    tmp = self.iou(box_predictions[i],
                                   box_predictions[j])
                    if tmp > self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accumulated_count += class_count

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load images
        """
        image_paths = glob.glob(folder_path + '/*.jpg')
        images = [cv2.imread(image) for image in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images
        """
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        processed_images_list = []
        image_shapes_list = []

        for image in images:
            image_shape = image.shape[0], image.shape[1]
            image_shapes_list.append(image_shape)

            dim = (input_w, input_h)
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

            processed_image = resized / 255
            processed_images_list.append(processed_image)

        processed_images = np.array(processed_images_list)
        image_shapes = np.array(image_shapes_list)

        return processed_images, image_shapes
