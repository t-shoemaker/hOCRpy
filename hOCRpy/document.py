#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .page import hOCR
import os
from PIL import Image
from typing import Union

class Document:

    def __init__(
        self,
        pages: Union[str, list],
        tesseract_output: bool=True
    ):
        if not isinstance(pages, str) and not isinstance(pages, list):
            raise ValueError("Invalid input. Use a path or list")

        # TODO: figure out how to parse multipage hOCR docs
        if isinstance(pages, str):
            self.pages = hOCR(pages, tesseract_output)
            self.num_pages = 1
        else:
            if all(isinstance(p, str) for p in pages):
                self.pages = [hOCR(p, tesseract_output) for p in pages]
            elif all(isinstance(p, hOCR) for p in pages):
                self.pages = [hocr for hocr in pages]
            else:
                raise ValueError("List of pages is invalid")
            self.num_pages = len(pages)

    def contact_sheet(
        self,
        num_col: int=2,
        width: int=595,
        height: int=842,
        margin: list=[5,5,5,5],
        padding: int=1,
        page_type: str='structure',
        return_image: bool=False,
        **kwargs
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
            ValueError if the number of columns exceeds the number of images
            ValueError if there is an invalid page_type
        """
        if num_col > self.num_pages:
            raise ValueError("Number of columns exceeds the number of pages")

        # Render each of the images
        if page_type == 'page':
            imgs = [
                p.show_page(return_image=True, **kwargs) for p in self.pages
            ]
        elif page_type == 'structure':
            imgs = [
                p.show_structure(return_image=True, **kwargs) for p in
                self.pages
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
            num_col * width + sheet_w + pad_w, num_row * height + sheet_w + pad_h
        )
        # Create the sheet
        output = Image.new('RGB', size=size, color='white')

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

