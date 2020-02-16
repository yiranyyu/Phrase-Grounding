def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def intersect(bbox1, bbox2):
    left = bbox1[0] if bbox1[0] > bbox2[0] else bbox2[0]
    top = bbox1[1] if bbox1[1] > bbox2[1] else bbox2[1]
    right = bbox1[2] if bbox1[2] < bbox2[2] else bbox2[2]
    bottom = bbox1[3] if bbox1[3] < bbox2[3] else bbox2[3]
    if left > right or top > bottom:
        return [0, 0, 0, 0]

    return [left, top, right, bottom]


def iou(bbox1, bbox2):
    area1 = area(bbox1)
    area2 = area(bbox2)
    intersection = intersect(bbox1, bbox2)
    area12 = area(intersection)
    return area12 / (area1 + area2 - area12)
