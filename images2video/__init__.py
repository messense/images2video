# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import cv2

from .effects import OriginalEffect


DEFAULT_FPS = 20
DEFAULT_FRAME_SIZE = (400, 400)
DEFAULT_VIDEO_SECONDS = 9


class ImagesToVideo(object):

    def __init__(self, filename, fourcc=None, fps=DEFAULT_FPS,
                 frame_size=DEFAULT_FRAME_SIZE, is_color=1,
                 seconds=DEFAULT_VIDEO_SECONDS):
        fourcc = fourcc or cv2.cv.CV_FOURCC(b'F', b'M', b'P', b'4')
        self.frame_count = fps * seconds
        self.frame_size = frame_size
        self.images = []

        self.video_writer = cv2.VideoWriter(
            filename=filename,
            fourcc=fourcc,
            fps=fps,
            frameSize=frame_size,
            isColor=is_color
        )

    def add_image(self, image, effect_class=None, **kwargs):
        effect_class = effect_class or OriginalEffect
        self.images.append((image, effect_class, kwargs))

    def generate(self):
        image_count = len(self.images)
        frames_per_image = self.frame_count / image_count

        for item in iter(self.images):
            filename = item[0]
            effect_class = item[1]
            options = item[2]
            image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_UNCHANGED)
            frames = effect_class(
                image,
                frames_per_image,
                self.frame_size,
                **options
            ).apply()
            for frame in frames:
                self.video_writer.write(frame)
        self.video_writer.release()
