# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import cv2
import numpy


EFFECTS = {}

DEFAULT_STATIC_FRAMES = 20


def register_effect(name):

    def register(cls):
        EFFECTS[name] = cls
        return cls
    return register


class FrameEffect(object):

    def __init__(self, image, frame_count, frame_size, bounce=False, **kwargs):
        self.image = image
        self.frame_size = frame_size
        self.bounce = bounce
        self.options = kwargs or {}
        self.frames = []
        self.static_frames = kwargs.get('static_frames', DEFAULT_STATIC_FRAMES)
        self.frame_count = frame_count - self.static_frames

    def apply(self):
        """Apply filter, return frames"""
        raise NotImplementedError()

    def _apply_static_frames(self):
        if self.static_frames <= 0:
            return
        for i in range(self.static_frames):
            self.frames.append(self.image.copy())


@register_effect('original')
class OriginalEffect(FrameEffect):

    def apply(self):
        for i in range(self.frame_count):
            self.frames.append(self.image)
        return self.frames


@register_effect('fadein')
class FadeInEffect(FrameEffect):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        alpha_per_frame = 0.7 / self.frame_count
        img = self.image.copy()
        img[0:width, 0:height] = numpy.uint8([255, 255, 255])

        for i in range(self.frame_count):
            alpha = alpha_per_frame * i + 0.3
            beta = 1.0 - alpha
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)

        self._apply_static_frames()
        return self.frames


@register_effect('fadeout')
class FadeOutEffect(FrameEffect):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        beta_per_frame = 0.3 / self.frame_count
        img = self.image.copy()
        img[0:width, 0:height] = numpy.uint8([255, 255, 255])

        for i in range(self.frame_count):
            beta = beta_per_frame * i + 0.3
            alpha = 1.0 - beta
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)

        self._apply_static_frames()
        return self.frames


@register_effect('fadeinout')
class FadeInOutEffect(FrameEffect):

    def apply(self):
        width = self.frame_size[0]
        height = self.frame_size[1]
        img = self.image.copy()
        img[0:width, 0:height] = numpy.uint8([255, 255, 255])

        frame_count = self.frame_count / 2
        alpha_per_frame = 0.7 / frame_count

        for i in range(frame_count):
            alpha = alpha_per_frame * i + 0.3
            beta = 1.0 - alpha
            frame = cv2.addWeighted(self.image, alpha, img, beta, 0)
            self.frames.append(frame)

        self._apply_static_frames()

        frame_left = self.frame_count - frame_count
        if frame_left > 0:
            beta_per_frame = 0.3 / frame_left
            for i in range(frame_left):
                beta = beta_per_frame * i + 0.3
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position = start + delta * i - width
            if position < 0:
                matrix = numpy.float32([
                    [1, 0, position],
                    [0, 1, 0]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, - delta * i],
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position = width - delta * i - start
            if position > 0:
                matrix = numpy.float32([
                    [1, 0, position],
                    [0, 1, 0]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, delta * i],
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position = start + delta * i - height
            if position < 0:
                matrix = numpy.float32([
                    [1, 0, 0],
                    [0, 1, position]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, - delta * i]
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position = height - delta * i - start
            if position > 0:
                matrix = numpy.float32([
                    [1, 0, 0],
                    [0, 1, position]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, 0],
                [0, 1, delta * i]
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position_x = delta_x * i - width + start
            position_y = delta_y * i - height + start
            if position_x < 0 and position_y < 0:
                matrix = numpy.float32([
                    [1, 0, position_x],
                    [0, 1, position_y]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, - delta_x * i],
                [0, 1, - delta_y * i]
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position_x = delta_x * i - width + start
            position_y = height - delta_y * i - start
            if position_x < 0 and position_y > 0:
                matrix = numpy.float32([
                    [1, 0, position_x],
                    [0, 1, position_y]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, - delta_x * i],
                [0, 1, delta_y * i]
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position_x = width - delta_x * i - start
            position_y = delta_y * i - height + start
            if position_x > 0 and position_y < 0:
                matrix = numpy.float32([
                    [1, 0, position_x],
                    [0, 1, position_y]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i],
                [0, 1, - delta_y * i]
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
        start = self.options.get('start', 0)

        for i in range(frame_count):
            position_x = width - delta_x * i - start
            position_y = height - delta_y * i - start
            if position_x > 0 and position_y > 0:
                matrix = numpy.float32([
                    [1, 0, position_x],
                    [0, 1, position_y]
                ])
                frame = cv2.warpAffine(self.image, matrix, self.frame_size)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        for i in range(self.frame_count - frame_count):
            matrix = numpy.float32([
                [1, 0, delta_x * i],
                [0, 1, delta_y * i]
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
            img[x:x + small_width, y:y + small_height] = small
            self.frames.append(img)

        self._apply_static_frames()

        frame_left = self.frame_count - frame_count
        for i in range(frame_left):
            img = self.image.copy()
            fx = fy = (percentage * (frame_left - i)) / 100.0
            if fx < 1:
                img[0:width, 0:height] = numpy.uint8([0, 0, 0])
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
                img[x:x + small_width, y:y + small_height] = small
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
            img[x + crop_width + 1:width, 0:height] = black_px
            img[x:x + crop_width + 1, 0:y] = black_px
            img[x:x + crop_width + 1, y + crop_height + 1:height] = black_px
            self.frames.append(img)

        self._apply_static_frames()

        frame_left = self.frame_count - frame_count
        for i in range(frame_left):
            crop_width = rate_x * (frame_left - i - 1)
            crop_height = rate_y * (frame_left - i - 1)
            x = (width - crop_width) / 2
            y = (height - crop_height) / 2
            img = self.image.copy()
            black_px = numpy.uint8([0, 0, 0])
            img[0:x, 0:height] = black_px
            img[x + crop_width + 1:width, 0:height] = black_px
            img[x:x + crop_width + 1, 0:y] = black_px
            img[x:x + crop_width + 1, y + crop_height + 1:height] = black_px
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

        self._apply_static_frames()

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
        from PIL import Image, ImageFilter

        frame_count = self.frame_count
        if self.bounce:
            frame_count /= 2
        radius = self.options.get('radius', 5)
        radius_per_frame = float(radius) / frame_count

        for i in range(frame_count):
            blur_radius = int(radius - radius_per_frame * i)
            if blur_radius > 0:
                img = Image.fromarray(self.image)
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                frame = numpy.array(img, dtype=numpy.uint8)
            else:
                frame = self.image.copy()
            self.frames.append(frame)

        self._apply_static_frames()

        frame_left = self.frame_count - frame_count
        if frame_left > 0:
            radius_per_frame = float(radius) / frame_left
            for i in range(frame_left):
                blur_radius = int(radius_per_frame * i)
                if blur_radius > 0:
                    img = Image.fromarray(self.image)
                    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                    frame = numpy.array(img, dtype=numpy.uint8)
                else:
                    frame = self.image.copy()
                self.frames.append(frame)

        return self.frames
