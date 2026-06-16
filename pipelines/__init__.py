# mangascour/pipelines/__init__.py

from .text_remove import TextRemovePipeline
from .manga_clean import MangaCleanPipeline

__all__ = ["TextRemovePipeline", "MangaCleanPipeline"]
