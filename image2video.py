# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import copy
import cv


DEFAULT_FPS = 20
DEFAULT_FRAME_SIZE = (400, 400)
DEFAULT_VIDEO_SECONDS = 9


def add_alpha_channel(image, alpha=255):
    for i in range(image.width):
        for j in range(image.height):
            channels = cv.Get2D(image, j, i)
            channels[3] = alpha
            cv.Set2D(image, j, i, channels)


class FrameFilter(object):

    def __init__(self, image, frame_count):
        self.image = image
        self.frame_count = frame_count
        self.frames = []

    def apply(self):
        """Apply filter, return frames"""
        raise NotImplementedError()


class OriginalFilter(FrameFilter):

    def apply(self):
        for i in range(self.frame_count):
            self.frames.append(self.image)
        return self.frames


class FadeInFilter(FrameFilter):

    def apply(self):
        pass


class FadeOutFilter(FrameFilter):

    def apply(self):
        pass


class FadeInOutFilter(FrameFilter):

    def apply(self):
        pass


class LeftToRightFilter(FrameFilter):

    def apply(self):
        pass


class RightToLeftFilter(FrameFilter):

    def apply(self):
        pass


class TopToBottomFilter(FrameFilter):

    def apply(self):
        pass


class BottomToTopFilter(FrameFilter):

    def apply(self):
        pass


class LeftTopToRightBottomFilter(FrameFilter):

    def apply(self):
        pass


class LeftBottomToRightTopFilter(FrameFilter):

    def apply(self):
        pass


class RightTopToLeftBottomFilter(FrameFilter):

    def apply(self):
        pass


class RightBottomToLeftTopFilter(FrameFilter):

    def apply(self):
        pass


class ResizeFilter(FrameFilter):

    def apply(self):
        pass


class CropFilter(FrameFilter):

    def apply(self):
        pass


class ImageToVideo(object):

    def __init__(self, filename, fourcc=None, fps=DEFAULT_FPS,
                 frame_size=DEFAULT_FRAME_SIZE, is_color=1,
                 seconds=DEFAULT_VIDEO_SECONDS):
        fourcc = fourcc or cv.CV_FOURCC(b'X', b'V', b'I', b'D')
        self.frame_count = fps * seconds
        self.images = []

        self.video_writer = cv.CreateVideoWriter(
            filename=filename,
            fourcc=fourcc,
            fps=fps,
            frame_size=frame_size,
            is_color=is_color
        )

    def add_image(self, image, filter_class=None):
        filter_class = filter_class or OriginalFilter
        self.images.append((image, filter_class))

    def generate(self):
        image_count = len(self.images)
        frames_per_image = self.frame_count / image_count

        for item in iter(self.images):
            filename = item[0]
            filter_class = item[1]
            image = cv.LoadImage(filename)
            frames = filter_class(image, frames_per_image).apply()
            for frame in frames:
                cv.WriteFrame(self.video_writer, frame)
        del self.video_writer  # save
