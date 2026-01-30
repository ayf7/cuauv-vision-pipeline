from vision.core.handlers import HandlerBase
import numpy as np

class StubHandler(HandlerBase):

    def process(self,
                direction: str,
                img: np.ndarray,
                *args, **kwargs
                ):
        pass