import torch

class Selector:

    def __init__(self, selected_labels, selection_size, 
                 label_multipliers=[]):
        """
        PARAMS:
            selected_labels: List of COCO label strings representing
                             the object classes that will be counted
                             for the selection.
            selection_size: Number of images to select
            label_multipliers: List of multipliers for object classes
        """

        # Initialize YOLOv5 detection model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.model.conf = 0.15
        self.model.iou = 0.45

        self.selected_labels = selected_labels

        self.reset(selection_size)
    

    def reset(self, selection_size, label_multipliers=[]):
        """
        PARAMS:
            selection_size: Number of images to select
            label_multipliers: List of multipliers for rare classes
        """

        count_selection_size = int(selection_size/2)
        if selection_size % 2:
            count_selection_size += 1

        # Reset lists for image infromation saving
        self.selected_image_counts = [0] * count_selection_size
        self.count_image_scores = [2.0] * count_selection_size
        self.count_selected_images = [None] * count_selection_size
        self.lowest_count = 0
        self.lowest_count_index = 0

        self.selected_image_scores = [2.0] * int(selection_size/2)
        self.score_selected_images = [None] * int(selection_size/2)
        self.highest_score = 2.0
        self.highest_score_index = 0

        # Generate dummy label multipliers if not given
        if not label_multipliers:
            label_multipliers = [1] * len(self.selected_labels)
        
        self.multipliers = {}
        for i, label in enumerate(self.selected_labels):
            self.multipliers[label] = label_multipliers[i]

    def get_selected(self):
        """
            RETURNS:
                List of selected image identifiers
        """
        return list(self.count_selected_images + 
                    self.score_selected_images)[:]

    def do_selection(self, image, image_name):
        """
            This method is used to do selection on images. The method 
            saves selected image identifiers to class members. The 
            selected images can be queried with get_selected() method. 
            The reset() method canbe used to restart the selection 
            done by this method.

            PARAMS:
                image: Next image to process as numpy array in RBG format
                image_name: Image identifier as string
        """
        # Do inference with YOLOv5
        results = self.model([image], size=640)

        total_count = 0
        average_score = 0

        names = results.names
        # Loop through detections and calculate count and score
        for i in range(results.n):
            for j in range(len(results.xyxy[i])):

                label = names[int(results.xyxy[i][j][-1])]
                score = results.xyxy[i][j][-2].item()

                if label in self.selected_labels:
                    total_count += 1 * self.multipliers[label]
                    average_score += score

        # Calculate average score
        if total_count > 0:
            average_score /= total_count
        else:
            average_score = 1.0

        if total_count > self.lowest_count:

            # Copy the score and name of the now replaced image
            replaced_image_name = self.count_selected_images[self.lowest_count_index]
            replaced_image_score = self.count_image_scores[self.lowest_count_index]

            # Save the new images information
            self.selected_image_counts[self.lowest_count_index] = total_count
            self.count_selected_images[self.lowest_count_index] = image_name
            self.count_image_scores[self.lowest_count_index] = average_score

            # Get new lowest count and list index
            self.lowest_count_index = self.count_selected.index(min(self.count_selected))
            self.lowest_count = self.count_selected[self.lowest_count_index]

            # Set the replaced image's score and name for score 
            # selection check
            average_score = replaced_image_score
            image_name = replaced_image_name

        if average_score < self.highest_score:

            # Save the new images information
            self.selected_image_scores[self.highest_score_index] = average_score
            self.score_selected_images[self.highest_score_index] = image_name

            # Get new highest score and list index
            self.highest_score_index = self.selected_image_scores.index(max(self.selected_image_scores))
            self.highest_score = self.selected_image_scores[self.highest_score_index]


