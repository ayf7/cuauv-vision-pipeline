#!/usr/bin/env python3
from vision.core import tuners
from vision.utils.draw import draw_text
from vision.core.base import ModuleBase, sources

module_tuners = [
    tuners.DoubleTuner('text_size', 5.0, 0.0, 20.0),
    tuners.IntTuner('text_thickness', 3, 1, 10)
]

class Hello(ModuleBase):

    @sources("zed[forward]")
    def process_img(self, image1):
        draw_text(image1, "Hello CUAUV!", (100, 200),
                  self.tuners["text_size"],
                  color=(255, 255, 255),
                  thickness=self.tuners["text_thickness"])
        self.post("hello", image1)

if __name__ == '__main__':
    Hello(["zed"], module_tuners)()