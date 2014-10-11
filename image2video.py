# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import cv2
import numpy


DEFAULT_FPS = 20
DEFAULT_FRAME_SIZE = (400, 400)
DEFAULT_VIDEO_SECONDS = 9


def add_alpha_channel(image, alpha=255):
    width = image.shape[0]
    height = image.shape[1]
    tmp_image = []
    for i in range(width):
        for j in range(height):
            tmp = numpy.append(image[i][j], alpha)
            tmp_image.append(tmp)
    img = numpy.array(tmp_image)
    img.resize((width, height, 4))
    return img


class FrameFilter(object):

    def __init__(self, image, frame_count, frame_size):
        self.image = image
        self.frame_count = frame_count
        self.frame_size = frame_size
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
        width = self.frame_size[0]
        delta = float(width) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, delta * i - width],
                [0, 1, 0]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class RightToLeftFilter(FrameFilter):

    def apply(self):
        width = self.frame_size[0]
        delta = float(width) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta * i],
                [0, 1, 0]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class TopToBottomFilter(FrameFilter):

    def apply(self):
        height = self.frame_size[1]
        delta = float(height) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, delta * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class BottomToTopFilter(FrameFilter):

    def apply(self):
        height = self.frame_size[1]
        delta = float(height) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, height - delta * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class LeftTopToRightBottomFilter(FrameFilter):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / self.frame_count
        delta_y = float(height) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i - width],
                [0, 1, delta_y * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class LeftBottomToRightTopFilter(FrameFilter):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / self.frame_count
        delta_y = float(height) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i - width],
                [0, 1, height - delta_y * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class RightTopToLeftBottomFilter(FrameFilter):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / self.frame_count
        delta_y = float(height) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta_x * i],
                [0, 1, delta_y * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class RightBottomToLeftTopFilter(FrameFilter):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / self.frame_count
        delta_y = float(height) / self.frame_count
        for i in range(self.frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta_x * i],
                [0, 1, height - delta_y * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)
        return self.frames


class ResizeFilter(FrameFilter):

    def apply(self):
        # FIXME: Resize incorrect
        width = self.frame_size[0]
        height = self.frame_size[1]
        percentage = 100.0 / self.frame_count
        for i in range(self.frame_count):
            fx = fy = (percentage * (i + 1)) / 100.0
            frame = cv2.resize(
                self.image,
                (0, 0),
                fx=fx,
                fy=fy
            )
            self.frames.append(frame)
        return self.frames


class CropFilter(FrameFilter):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        rate_x = width / self.frame_count
        rate_y = height / self.frame_count
        for i in range(self.frame_count):
            crop_width = rate_x * (i + 1)
            crop_height = rate_y * (i + 1)
            x = (width - crop_width) / 2
            y = (height - crop_height) / 2
            img = self.image.copy()
            black_px = numpy.uint8([0, 0, 0])
            img[0:x, 0:height] = black_px
            img[x+crop_width+1:width, 0:height] = black_px
            img[x:x+crop_width+1, 0:y] = black_px
            img[x:x+crop_width+1, y+crop_height+1:height] = black_px
            self.frames.append(img)
        return self.frames


class ImageToVideo(object):

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

    def add_image(self, image, filter_class=None):
        filter_class = filter_class or OriginalFilter
        self.images.append((image, filter_class))

    def generate(self):
        image_count = len(self.images)
        frames_per_image = self.frame_count / image_count

        for item in iter(self.images):
            filename = item[0]
            filter_class = item[1]
            image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_UNCHANGED)
            frames = filter_class(
                image,
                frames_per_image,
                self.frame_size
            ).apply()
            for frame in frames:
                self.video_writer.write(frame)
        self.video_writer.release()


if __name__ == '__main__':
    converter = ImageToVideo('test.avi')
    converter.add_image('1.jpg', CropFilter)
    # converter.add_image('1.jpg', LeftBottomToRightTopFilter)
    # converter.add_image('1.jpg', RightTopToLeftBottomFilter)
    # converter.add_image('1.jpg', RightBottomToLeftTopFilter)
    converter.generate()
