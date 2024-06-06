#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .page import hOCR
import os
from PIL import Image
from typing import List, Union


class Document:
    def __init__(
        self,
        pages: Union[str, List[Union[str, hOCR]]],
        tesseract_output: bool = True,
    ):
        """Initialize the document.

        Parameters
        ----------
        pages
            The source from which to build the document. This may either be a
            single, multi-page hOCR file, a list of path strings to individual
            hOCR files, or hOCR objects
        tesseract_output
            Whether the hOCR has been produced by Tesseract
        """
        if not isinstance(pages, str) and not isinstance(pages, list):
            raise ValueError("Invalid input. Use a path or list")

        # TODO: figure out how to parse multipage hOCR docs
        if isinstance(pages, str):
            self._pages = hOCR(pages, tesseract_output)
            self.num_pages = 1
        else:
            if all(isinstance(p, str) for p in pages):
                self._pages = tuple(hOCR(p, tesseract_output) for p in pages)
            elif all(isinstance(p, hOCR) for p in pages):
                self._pages = tuple(hocr for hocr in pages)
            else:
                raise ValueError("List of pages is invalid")
            self.num_pages = len(pages)

    def __getitem__(self, idx) -> hOCR:
        """When indexed, return the associated page."""
        return self._pages[idx]

    def __repr__(self) -> str:
        """Object representation."""
        return f"hOCR Document with {self.num_pages} pages"

    @property
    def num_tokens(self) -> int:
        """Total number of tokens."""
        return sum(p.num_tokens for p in self._pages)

    @property
    def text(self) -> str:
        """Return a plaintext blob of all the document tokens."""
        return " ".join(p.text for p in self._pages)

    @property
    def scores(self) -> List[float]:
        """Return the confidence scores for all tokens."""
        _scores = [p.scores for p in self._pages]

        return [s for p in _scores for s in p]

    def contact_sheet(
        self,
        num_col: int = 2,
        width: int = 595,
        height: int = 842,
        margin: list = [5, 5, 5, 5],
        padding: int = 1,
        page_type: str = "structure",
        return_image: bool = False,
        **kwargs,
    ) -> Image.Image:
        """Create a contact sheet of the whole document.

        The default width and height settings are for A4 paper. This is loosely
        based on
        https://code.activestate.com/recipes/412982-use-pil-to-make-a-contact-sheet-montage-of-images/

        Parameters
        ----------
        num_col
            Number of columns
        width
            Width of each page
        height
            Height of each page
        margin
            Margins around the sheet. Order is left, right, top bottom
        padding
            Space between pages on the sheet
        page_type
            Whether to return actual pages or the page structure. See the hOCR
            class for more info
        return_image
            Whether to render or return the image
        kwargs
            Optional arguments for styling the pages

        Returns
        -------
        sheet
            A contact sheet view

        Raises
        ------
        AssertionError
            If the number of columns exceeds the number of images
        ValueError
            If there is an invalid page_type
        """
        # Ensure that the columns do not exceed the images
        assert num_col < self.num_pages, "Number of columns exceeds page count"

        # Render each of the images
        if page_type == "page":
            imgs = [
                p.show_page(return_image=True, **kwargs) for p in self._pages
            ]
        elif page_type == "structure":
            imgs = [
                p.show_structure(return_image=True, **kwargs)
                for p in self._pages
            ]
        else:
            raise ValueError("Valid page types are `page` and `structure`")

        # Resize images
        imgs = [i.resize((width, height)) for i in imgs]
        # Calculate the number of rows
        num_row = num_col % self.num_pages

        # Get the dimensions of the whole contact sheet
        sheet_w, sheet_h = margin[0] + margin[1], margin[2] + margin[3]
        pad_w, pad_h = (num_col - 1) * padding, (num_row - 1) * padding
        size = (
            num_col * width + sheet_w + pad_w,
            num_row * height + sheet_w + pad_h,
        )
        # Create the sheet
        output = Image.new("RGB", size=size, color="white")

        # Now do a row- and column-wise iteration and find where each image
        # should be positioned on the sheet
        for row in range(num_row):
            for col in range(num_col):
                left = margin[0] + col * (width + padding)
                right = left + width
                upper = margin[2] + row * (height + padding)
                lower = upper + height
                bbox = (left, upper, right, lower)
                # Pop images out one by one
                if imgs:
                    img = imgs.pop(0)
                else:
                    break
                output.paste(img, bbox)

        if not return_image:
            output.show()
        else:
            return output
