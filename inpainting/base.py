from abc import ABC, abstractmethod


class Inpainter(ABC):

    @abstractmethod
    def run(self, image, mask):
        pass