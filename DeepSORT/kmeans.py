import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        min_w_matrix = np.minimum(boxes[:, 0][:, np.newaxis], clusters[:, 0])
        min_h_matrix = np.minimum(boxes[:, 1][:, np.newaxis], clusters[:, 1])
        inter_area = min_w_matrix * min_h_matrix

        result = inter_area / (box_area[:, np.newaxis] + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean(np.max(self.iou(boxes, clusters), axis=1))
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest
        return clusters

    def result2txt(self, data):
        with open("yolo_anchors.txt", 'w') as f:
            rows = np.shape(data)[0]
            for i in range(rows):
                x_y = f"{data[i][0]},{data[i][1]}"
                if i != 0:
                    x_y = ", " + x_y
                f.write(x_y)

    def txt2boxes(self):
        with open(self.filename, 'r') as f:
            dataSet = []
            for line in f:
                infos = line.split(" ")
                for info in infos[1:]:
                    width, height = [int(coord) for coord in info.split(",")[2:4]]
                    dataSet.append([width - int(info.split(",")[0]), height - int(info.split(",")[1])])
        return np.array(dataSet)

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
