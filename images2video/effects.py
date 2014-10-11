# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import cv2
import numpy


EFFECTS = {}


def register_effect(name):

    def register(cls):
        EFFECTS[name] = cls
        return cls
    return register


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


@register_effect('original')
class OriginalEffect(FrameEffect):

    def apply(self):
        for i in range(self.frame_count):
            self.frames.append(self.image)
        return self.frames


class FadeInEffect(FrameEffect):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        alpha_per_frame = 1.0 / self.frame_count
        img = self.image.copy()
        img[0:width, 0:height] = numpy.uint8([255, 255, 255])

        for i in range(self.frame_count):
            alpha = alpha_per_frame * i
            beta = 1.0 - alpha
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)
        return self.frames


class FadeOutEffect(FrameEffect):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        beta_per_frame = 1.0 / self.frame_count
        img = self.image.copy()
        img[0:width, 0:height] = numpy.uint8([255, 255, 255])

        for i in range(self.frame_count):
            beta = beta_per_frame * i
            alpha = 1.0 - beta
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)
        return self.frames


class FadeInOutEffect(FrameEffect):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        img = self.image.copy()
        img[0:width, 0:height] = numpy.uint8([255, 255, 255])

        frame_count = self.frame_count / 2
        alpha_per_frame = 1.0 / frame_count

        for i in range(frame_count):
            alpha = alpha_per_frame * i
            beta = 1.0 - alpha
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)

        frame_left = self.frame_count - frame_count
        beta_per_frame = 1.0 / frame_left
        for i in range(frame_left):
            beta = beta_per_frame * i
            alpha = 1.0 - beta
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)

        return self.frames


@register_effect('left2right')
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


@register_effect('right2left')
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


@register_effect('top2bottom')
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


@register_effect('bottom2top')
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


@register_effect('lefttop2rightbottom')
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


@register_effect('leftbottom2righttop')
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


@register_effect('righttop2leftbottom')
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


@register_effect('rightbottom2lefttop')
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


@register_effect('resize')
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


@register_effect('crop')
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


@register_effect('rotation')
class RotationEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        width = self.frame_size[0]
        height = self.frame_size[1]
        percentage = 100.0 / frame_count
        angle_per_frame = 360.0 / frame_count

        for i in range(frame_count):
            scale = (percentage * (i + 1)) / 100.0
            angle = i * angle_per_frame
            matrix = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                angle,
                scale
            )
            frame = cv2.warpAffine(self.image, matrix, (width, height))
            self.frames.append(frame)

        frame_left = self.frame_count - frame_count
        for i in range(frame_left):
            scale = (percentage * (frame_left - i)) / 100.0
            angle = - i * angle_per_frame
            matrix = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                angle,
                scale
            )
            frame = cv2.warpAffine(self.image, matrix, (width, height))
            self.frames.append(frame)

        return self.frames


@register_effect('blur')
class BlurEffect(FrameEffect):

    def apply(self):
        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        radius = self.options.get('radius', 5)
        radius_per_frame = float(radius) / frame_count

        for i in range(frame_count):
            blur_radius = int(radius - radius_per_frame * i)
            if blur_radius > 0:
                kernel = (blur_radius, blur_radius)
                frame = cv2.blur(self.image, kernel)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        frame_left = self.frame_count - frame_count
        radius_per_frame = float(radius) / frame_left
        for i in range(frame_left):
            blur_radius = int(radius_per_frame * i)
            if blur_radius > 0:
                kernel = (blur_radius, blur_radius)
                frame = cv2.blur(self.image, kernel)
            else:
                frame = self.image.copy()
            self.frames.append(frame)
        return self.frames
