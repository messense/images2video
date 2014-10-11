# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import cv2
import numpy


DEFAULT_FPS = 20
DEFAULT_FRAME_SIZE = (400, 400)
DEFAULT_VIDEO_SECONDS = 9


class FrameEffect(object):

    def __init__(self, image, frame_count, frame_size, bounce=False, **kwargs):
        self.image = image
        self.frame_count = frame_count
        self.frame_size = frame_size
        self.bounce = bounce
        self.options = kwargs or {}
        self.frames = []

    def apply(self):
        """Apply filter, return frames"""
        raise NotImplementedError()


class OriginalEffect(FrameEffect):

    def apply(self):
        for i in range(self.frame_count):
            self.frames.append(self.image)
        return self.frames


class FadeInEffect(FrameEffect):

    def apply(self):
        pass


class FadeOutEffect(FrameEffect):

    def apply(self):
        pass


class FadeInOutEffect(FrameEffect):

    def apply(self):
        pass


class LeftToRightEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        delta = float(width) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, delta * i - width],
                [0, 1, 0]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta * i],
                [0, 1, 0]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class RightToLeftEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        delta = float(width) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta * i],
                [0, 1, 0]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, delta * i - width],
                [0, 1, 0]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class TopToBottomEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        height = self.frame_size[1]
        delta = float(height) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, delta * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, height - delta * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class BottomToTopEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        height = self.frame_size[1]
        delta = float(height) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, height - delta * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, delta * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class LeftTopToRightBottomEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / frame_count
        delta_y = float(height) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i - width],
                [0, 1, delta_y * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta_x * i],
                [0, 1, height - delta_y * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class LeftBottomToRightTopEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / frame_count
        delta_y = float(height) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i - width],
                [0, 1, height - delta_y * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta_x * i],
                [0, 1, delta_y * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class RightTopToLeftBottomEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / frame_count
        delta_y = float(height) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta_x * i],
                [0, 1, delta_y * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i - width],
                [0, 1, height - delta_y * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class RightBottomToLeftTopEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        delta_x = float(width) / frame_count
        delta_y = float(height) / frame_count

        for i in range(frame_count):
            matrix = numpy.float32([
                [1, 0, width - delta_x * i],
                [0, 1, height - delta_y * i]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i - width],
                [0, 1, delta_y * i - height]
            ])
            frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            self.frames.append(frame)

        return self.frames


class ResizeEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        percentage = 100.0 / frame_count

        for i in range(frame_count):
            img = self.image.copy()
            img[0:width, 0:height] = numpy.uint8([0, 0, 0])
            fx = fy = (percentage * (i + 1)) / 100.0
            small = cv2.resize(
                self.image,
                (0, 0),
                fx=fx,
                fy=fy
            )
            small_width = small.shape[0]
            small_height = small.shape[1]
            x = (width - small_width) / 2
            y = (height - small_height) / 2
            img[x:x+small_width, y:y+small_height] = small
            self.frames.append(img)

        frame_left = self.frame_count - frame_count
        for i in range(frame_left):
            img = self.image.copy()
            img[0:width, 0:height] = numpy.uint8([0, 0, 0])
            fx = fy = (percentage * (frame_left - i)) / 100.0
            small = cv2.resize(
                self.image,
                (0, 0),
                fx=fx,
                fy=fy
            )
            small_width = small.shape[0]
            small_height = small.shape[1]
            x = (width - small_width) / 2
            y = (height - small_height) / 2
            img[x:x+small_width, y:y+small_height] = small
            self.frames.append(img)

        return self.frames


class CropEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        rate_x = float(width) / frame_count
        rate_y = float(height) / frame_count

        for i in range(frame_count):
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

        frame_left = self.frame_count - frame_count
        for i in range(frame_left):
            crop_width = rate_x * (frame_left - i - 1)
            crop_height = rate_y * (frame_left - i - 1)
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


class RotationEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]


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


if __name__ == '__main__':
    converter = ImageToVideo('test.avi', fps=40)
    converter.add_image('1.jpg', ResizeEffect, bounce=True)
    converter.add_image('1.jpg', CropEffect, bounce=True)
    converter.add_image('1.jpg', RightTopToLeftBottomEffect, bounce=True)
    converter.add_image('1.jpg', RightBottomToLeftTopEffect, bounce=True)
    converter.generate()
