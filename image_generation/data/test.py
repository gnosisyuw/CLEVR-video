import os
import cv2

import skvideo.io

dir = '/home/wei/PycharmProjects/stacking/output/images/'
vs = os.listdir(dir)
vs.sort()

sname = dir + 'data.mp4'
writer = skvideo.io.FFmpegWriter(sname, inputdict={'-r': str(30), },
                                     outputdict={'-r': str(30), '-vcodec': 'libx264', '-crf': '23'})
for v in vs:
    frame = cv2.imread(dir + v)

    writer.writeFrame(frame[:, :, ::-1])
writer.close()