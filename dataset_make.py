import cv2
import numpy as np
import tfrecord


VIDEO_PATH = "TownCentreXVID.avi"
videoCapture = cv2.VideoCapture()
videoCapture.open(VIDEO_PATH)
length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
image_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
image_channel = 3

print(length, image_width, image_height)

writer = tfrecord.TFRecordWriter("train.tfrecord")
ret = True
while ret:
    img_batch_rgb = np.empty(
        shape=[0, image_channel, image_height, image_width],
        dtype=np.uint8
    )
    for i in range(2):
        ret, img = videoCapture.read()
        if ret:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_batch_rgb=np.append(img_batch_rgb, np.expand_dims(rgb_img.transpose(2, 0, 1), 0), axis=0)
        else:
            break

    for x in np.arange(0, image_height - 128 + 1, 100):
        for y in np.arange(0, image_width - 128 + 1, 100):
            # print("croping----")
            img_part = img_batch_rgb[:, :,
                                     int(x): int(x + 128),
                                     int(y): int(y + 128)]
            img_bytes = img_part.tobytes()

            writer.write({
                "image": (img_bytes, "byte"),
                "size": (128, "int"),
            })

writer.close()
videoCapture.release()
