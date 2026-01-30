#!/usr/bin/env python3
import numpy as np

import shm
from vision.core.base import ModuleBase
from vision.utils.draw import draw_circle

try:
    from vision.capture_sources.zed import ZED_MAX_DISTANCE, ZED_MIN_DISTANCE
except ImportError:
    ZED_MIN_DISTANCE = 0
    ZED_MAX_DISTANCE = 1


# TODO: add argument for current submission, cases for each submission
class Relay(ModuleBase):

    def denormalize(self, image, x, y):
        width, height = len(image[0]), len(image)
        new_x = int(x * width + width / 2) + 50  # normal is for some reason -50 off
        new_y = int(y * width + height / 2)

        new_x = max(0, min(new_x, width - 1))
        new_y = max(0, min(new_y, height - 1))

        return (new_x, new_y)

    def sample_patch(self, normals, center, radius=1):
        x, y = center
        h, w, _ = normals.shape
        x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
        patch = normals[y0:y1, x0:x1].reshape(-1, 3)
        return patch

    def filter_normals(self, vectors, std_threshold=2.0, min_value=10.0):
        vectors = vectors[np.all(vectors >= min_value, axis=1)]
        if len(vectors) == 0:
            return np.empty((0, 3))

        mean_vec = np.mean(vectors, axis=0)
        dists = np.linalg.norm(vectors - mean_vec, axis=1)
        threshold = np.mean(dists) + std_threshold * np.std(dists)
        return vectors[dists < threshold]

    def simple_average(self, vectors):
        if len(vectors) == 0:
            return np.array([0.0, 0.0, 0.0])  # fallback
        return np.mean(vectors, axis=0)

    def process(self, direction, image):
        if direction == "normal":
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        location = (shm.relay.point_x.get(), shm.relay.point_y.get())

        denormalized_location = self.denormalize(image, location[0], location[1])
        draw_circle(
            image, denormalized_location, radius=5, color=(0, 255, 0), thickness=3
        )

        keypoints = [denormalized_location]

        if shm.yolo_torpedoes_board.board_visible.get():
            board_group = shm.yolo_torpedoes_board

            # assumes square isn't tilted
            board_width = (
                self.denormalize(image, board_group.board_top_right_x.get(), 0)[0]
                - self.denormalize(image, board_group.board_top_left_x.get(), 0)[0]
            )
            board_height = (
                self.denormalize(image, 0, board_group.board_bottom_right_y.get())[1]
                - self.denormalize(image, 0, board_group.board_top_right_y.get())[1]
            )

            stretch = 1 / 6

            tl = (
                int(denormalized_location[0] - stretch * board_width),
                int(denormalized_location[1] - stretch * board_height),
            )
            tr = (
                int(denormalized_location[0] + stretch * board_width),
                int(denormalized_location[1] - stretch * board_height),
            )
            bl = (
                int(denormalized_location[0] - stretch * board_width),
                int(denormalized_location[1] + stretch * board_height),
            )
            br = (
                int(denormalized_location[0] + stretch * board_width),
                int(denormalized_location[1] + stretch * board_height),
            )

            draw_circle(image, tl, radius=10, color=(255, 0, 0), thickness=3)
            draw_circle(image, tr, radius=10, color=(255, 0, 0), thickness=3)
            draw_circle(image, bl, radius=10, color=(255, 0, 0), thickness=3)
            draw_circle(image, br, radius=10, color=(255, 0, 0), thickness=3)

            keypoints.append(tl)
            keypoints.append(tr)
            keypoints.append(bl)
            keypoints.append(br)

        final_normals = []
        for pt in keypoints:
            patch = self.sample_patch(image, pt, radius=2)
            cleaned = self.filter_normals(patch)
            avg_normal = self.simple_average(cleaned)
            final_normals.append(avg_normal)

        final_normals = np.array(final_normals)
        final_avg_normal = np.mean(final_normals, axis=0)

        print("normal vector: ", final_avg_normal)  # [DEBUG]
        shm.relay.normal_x_at_point.set(final_avg_normal[0])
        shm.relay.normal_y_at_point.set(final_avg_normal[1])
        shm.relay.normal_z_at_point.set(final_avg_normal[2])

        self.post(direction, image)


if __name__ == "__main__":
    Relay(["normal"])()
