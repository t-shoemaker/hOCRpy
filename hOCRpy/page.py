#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
import lxml.html as HT
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Union


class hOCR:
    def __init__(self, path: str, tesseract_output: bool = True):
        """Load and parse the hOCR.

        Parameters
        ----------
        path
            Path to the hOCR file
        tesseract_output
            Whether the hOCR has been produced by Tesseract

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file at {path}")

        if tesseract_output:
            self.tree = HT.parse(path)
        else:
            self.tree = ET.parse(path)

    def __repr__(self) -> str:
        x, y = self.dim
        output = f"{x}x{y} page with {self.num_tokens} tokens"

        return output

    @property
    def dim(self) -> Tuple[int]:
        """Find the dimensions of the page."""
        div = self.tree.find('.//div[@class="ocr_page"]')
        div = div.attrib["title"].split(";")
        coords = div[1].split()
        dim = [int(c) for c in coords[-2:]]

        return tuple(dim)

    @property
    def word_spans(self) -> Union[List[HT.HtmlElement], List[ET.Element]]:
        """Retrieve the word spans from the hOCR.

        Returns
        -------
        spans
            A list of XML or HTML generators
        """
        return self.tree.findall('.//span[@class="ocrx_word"]')

    @property
    def tokens(self) -> List[str]:
        """Retrieve tokens from each word span."""
        return [span.text for span in self.word_spans]

    @property
    def num_tokens(self) -> int:
        """Count the number of tokens on the page."""
        return len(self.word_spans)

    @property
    def text(self) -> str:
        """Return a plaintext blob of all the page tokens."""
        return " ".join(span.text for span in self.word_spans)

    @property
    def token_data(self) -> List[Tuple[List[int], float]]:
        """Retrieve bounding box and confidence score data for every token.

        Returns
        -------
        token_data
            A list of tuples, where the first element therein is a list of
            bounding box positions and the second is a confidence score
        """
        return [self._extract_data(span) for span in self.word_spans]

    @property
    def bboxes(self) -> List[int]:
        """Return the bounding boxes from the token data."""
        return [i[0] for i in self.token_data]

    @property
    def scores(self) -> List[float]:
        """Return the confidence scores from the token data."""
        return [i[1] for i in self.token_data]

    def _extract_data(
        self, div: Union[HT.HtmlElement, ET.Element]
    ) -> Tuple[List[int], float]:
        """Extract the bounding box and confidence score of a div.

        Parameters
        ----------
        div
            The div to extract

        Returns
        -------
        data
            A bounding box (list) and confidence score (float)
        """
        # Certain page structures do not have extra info in their titles, so
        # we check for this first
        div = div.attrib["title"]
        if ";" not in div:
            div = div.split()
            bbox = [int(b) for b in div[1:]]

            return bbox, None

        div = div.split(";")
        bbox, score = div[0], div[1]
        bbox = [int(b) for b in bbox.split()[1:]]

        # Check for a score and format accordingly
        if any("x_wconf" in e for e in div) == False:
            score = None
        else:
            # We convert to a float, 0-1
            score = score.strip().split()
            score = int(score[1]) / 100

        return bbox, score

    def text_to_column(self, labels: np.array) -> Dict[int, str]:
        """Given a list of indexed labels, assign a token to its respective
        label.

        Here, a label corresponds to what a k-means clustering analysis has
        determined to be a column.

        TODO: sort the columns

        Parameters
        ----------
        labels
            The output of a k-means clusterer

        Returns
        -------
        columns
            Tokens separated into each of their label groups
        """
        columns = defaultdict(list)
        for label, span in zip(labels, self.word_spans):
            columns[label].append(span.text)

        return columns

    def show_structure(
        self, which: str = "line", return_image: bool = False, **kwargs
    ) -> Image.Image:
        """Show a high-level view of page elements.

        Parameters
        ----------
        which
            The type of view to return
        return_image
            Whether to render or return the image
        kwargs
            Optional arguments for styling the page and bounding box color. See
            the PageImage class

        Returns
        -------
        structure
            The rendered page

        Raises
        ------
        ValueError
            If provided invalid options for `which`
        """
        OPTS = {
            "area": {"div": "div", "class": "ocr_carea"},
            "paragraph": {"div": "p", "class": "ocr_par"},
            "line": {"div": "span", "class": "ocr_line"},
            "token": {"div": "span", "class": "ocrx_word"},
        }
        if which not in OPTS:
            raise ValueError(f"Valid options: {', '.join(OPTS.keys())}")

        # Query the XPath and extract the bounding boxes
        divs = self.tree.findall(
            f".//{OPTS[which]['div']}[@class='{OPTS[which]['class']}']"
        )
        divs = [self._extract_data(div) for div in divs]
        bboxes = [div[0] for div in divs]

        # Make a page image
        fill = kwargs.get("fill", "black")
        bfill = kwargs.get("bfill", "white")
        img = PageImage(self.dim, bfill)
        # Call the structure renderer
        structure = img.make_structure(bboxes, fill=fill)

        if not return_image:
            structure.show()
        else:
            return structure

    def show_page(
        self, scale: bool = False, return_image: bool = False, **kwargs
    ) -> Image.Image:
        """Render the page with tokens.

        Parameters
        ----------
        scale
            Scale the font size for each token to fill its bounding box
        return_image
            Whether to render or return the image
        kwargs
            Optional arguments for styling the bounding boxes. See the
            PageImage class

        Returns
        -------
        page
            Rendered page
        """
        tokens = [span.text for span in self.word_spans]

        # Make a page image
        img = PageImage(self.dim, kwargs.get("bfill", "white"))

        # Call the page renderer
        page = img.make_page(
            tokens,
            self.bboxes,
            self.scores,
            outline=kwargs.get("outline", "black"),
            show_conf=kwargs.get("show_conf", False),
            scale=scale,
            use_font=kwargs.get("use_font", "Arial.ttf"),
            text_fill=kwargs.get("text_fill", "black"),
            align=kwargs.get("align", "center"),
            anchor=kwargs.get("anchor", "ms"),
        )

        if not return_image:
            page.show()
        else:
            return page


class PageImage:
    def __init__(self, size: int, color: str = "white"):
        """Initialize a new page

        Parameters
        ----------
        size
            Size of the image
        color
            Color
        """
        self.img = Image.new(mode="RGB", size=size, color=color)

    def make_structure(self, bboxes: list, fill: str = "black") -> Image.Image:
        """Show a high-level view of the hOCR divs.

        Parameters
        ----------
        bboxes
            The bounding box data
        fill
            Fill color

        Returns
        -------
        img
            Rendered page
        """
        draw = ImageDraw.Draw(self.img)
        for bbox in bboxes:
            draw.rectangle(bbox, fill=fill)

        return self.img

    def make_page(
        self,
        tokens: list,
        bboxes: list,
        scores: list,
        outline: Union[None, str] = "black",
        show_conf: bool = False,
        scale: bool = False,
        use_font: str = "Arial.ttf",
        text_fill: str = "black",
        align: str = "center",
        anchor: str = "ms",
    ) -> Image.Image:
        """Render a page.

        TODO: implement show_conf, which wll be an opacity value

        Parameters
        ----------
        tokens
            The tokens to draw
        bboxes
            The bounding boxes to use
        scores
            The confidence scores for each token
        outline
            Color of the bounding boxes, pass None for no bounding boxes
        show_conf
            Change the opacity of tokens to reflect their confidence scores
        scale
            Scale the font size for each token to fill its bounding box
        use_font
            Font to use (can be a path)
        text_fill
            Text color
        align
            Text alignment
        anchor
            Text anchor

        Returns
        -------
        img
            Rendered page
        """
        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype(use_font)

        for token, bbox, score in zip(tokens, bboxes, scores):
            draw.rectangle(bbox, outline=outline)

            if scale:
                font_size = self._scale_font(
                    token, bbox, font_size=16, use_font=use_font
                )
                font = ImageFont.truetype(use_font, font_size)

            draw.text(
                xy=bbox[:2],
                text=token,
                fill=text_fill,
                font=font,
                align=align,
                anchor=anchor,
            )

        return self.img

    def _scale_font(
        self,
        token: str,
        bbox: list,
        font_size: int = 16,
        use_font: str = "Arial.ttf",
    ) -> int:
        """For a given token, find the font size that best fills the bounding
        box area.

        Parameters
        ----------
        token
            Text to render
        bbox
            The bounding box to fill
        font_size
            The size to start at
        use_font
            Font to use

        Returns
        -------
        font_size
            Font size that best approximates the bounding box area
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        size = None
        font = ImageFont.truetype(use_font, font_size)
        while (size is None or (size[0] * size[1]) > area) and font_size > 0:
            font = ImageFont.truetype(use_font, font_size)
            size = font.getsize(token)
            font_size -= 1

        return font_size
