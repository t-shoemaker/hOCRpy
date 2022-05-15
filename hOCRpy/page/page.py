#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import lxml.html as HT
from PIL import Image, ImageDraw, ImageFont
import csv

class hOCR:

    def __init__(self, path, tesseract_output=True):
        """Load and parse the hOCR."""
        if tesseract_output:
            self.tree = HT.parse(path)
        else:
            self.tree = ET.parse(path)

    @property
    def dim(self):
        """Find the dimensions of the page."""
        div = self.tree.find('.//div[@class="ocr_page"]')
        div = div.attrib['title'].split(';')
        coords = div[1].split()
        dim = [int(c) for c in coords[-2:]]
        
        return tuple(dim)

    @property
    def word_spans(self):
        """Retrieve the word spans from the hOCR."""
        return self.tree.findall('.//span[@class="ocrx_word"]')

    @property
    def num_tokens(self):
        """Count the number of tokens on the page."""
        return len(self.word_spans)

    @property
    def text(self):
        """Return a plaintext blob of all the page tokens."""
        return ' '.join(span.text for span in self.word_spans)

    @property
    def token_data(self):
        """Retrieve bounding box and confidence score data for every token."""
        return [self._extract_data(span) for span in self.word_spans]

    @property
    def bboxes(self):
        """Return the bounding boxes from the token data."""
        return [i[0] for i in self.token_data]

    @property
    def scores(self):
        """Return the confidence scores from the token data."""
        return [i[1] for i in self.token_data]

    def _extract_data(self, div):
        """Extract the bounding box and confidence score of a div.

        :param div: The div to extract
        :type div: XML generator
        "returns: Bounding box (list) and score (int)
        :rtype: tup
        """
        # Certain page structures do not have extra info in their titles, so 
        # we check for this first
        div = div.attrib['title']
        if ';' not in div:
            div = div.split()
            bbox = [int(b) for b in div[1:]]
            
            return bbox, None

        div = div.split(';')
        bbox, score = div[0], div[1]
        bbox = [int(b) for b in bbox.split()[1:]]
        
        # Check for a score and format accordingly
        if any('x_wconf' in e for e in div) == False:
            score = None
        else:
            # We convert to a float, 0-1
            score = score.strip().split()
            score = int(score[1]) / 100

        return bbox, score

    def show_structure(
        self,
        which='line',
        fill='black',
        bfill='white',
        return_image=False
        ):
        """Show a high-level view of page elements.

        :param which: The type of view to return
        :type which: str
        :raises ValueError: Valid options for `which` are area, paragraph, line
        :param fill: Fill color
        :type fill: str
        :param bfill: Background color
        :type bfill: str
        :param return_image: Specify whether to render or return the image
        :type return_image: bool
        :returns: Rendered page
        :rtype: PIL image
        """
        OPTS = {
            'area': {'div': 'div', 'class': 'ocr_carea'},
            'paragraph': {'div': 'p', 'class': 'ocr_par'},
            'line': {'div': 'span', 'class': 'ocr_line'},
        }
        if which not in OPTS:
            raise ValueError(f"Valid options: {', '.join(OPTS.keys())}")

        # Query the XPath and extract the bounding boxes 
        divs = self.tree.findall(f".//{OPTS[which]['div']}[@class='{OPTS[which]['class']}']")
        divs = [self._extract_data(div) for div in divs]
        bboxes = [div[0] for div in divs]

        # Make a page image
        img = PageImage(self.dim, bfill)
        # Call the structure renderer
        structure = img.make_structure(bboxes, fill=fill)

        if not return_image:
            structure.show()
        else:
            return structure

    def show_page(
        self,
        outline='black',
        bfill='white',
        show_conf=False,
        scale=False,
        use_font='Arial.ttf',
        return_image=False
        ):
        """Render the page with tokens.

        :param outline: Color of the bounding boxes, pass None for no bounding box outline
        :type outline: str or None
        :param bfill: The background color of the page
        :type bfill: str
        :param show_conf: Change the opacity of tokens to reflect their confidence score
        :type show_conf: bool
        :param scale: Scale the font size for each token to fill the bounding box
        :type scale: bool
        :param use_font: Font to use (can be a path)
        :type use_font: str
        :param return_image: Specify whether to render or return the image
        :type return_image: bool
        :Returns: Rendered page
        :rtype: PIL image
        """
        tokens = [span.text for span in self.word_spans]
        
        # Make a page image
        img = PageImage(self.dim, bfill)

        # Call the page renderer
        page = img.make_page(
            tokens,
            self.bboxes,
            self.scores,
            outline,
            show_conf,
            scale,
            use_font
        )

        if not return_image:
            page.show()
        else:
            return page

class PageImage:

    def __init__(self, size, color='white'):
        self.img = Image.new(mode='RGB', size=size, color=color)
    
    def make_structure(self, bboxes, fill='black'):
        """Show a high-level view of the hOCR divs.

        :param bboxes: The bounding box data
        :type bboxes: list
        :param fill: Fill color
        :type fill: str
        :returns: Rendered page
        :rtype: PIL image
        """
        draw = ImageDraw.Draw(self.img)
        for bbox in bboxes:
            draw.rectangle(bbox, fill=fill)

        return self.img
    
    def make_page(
        self,
        tokens,
        bboxes,
        scores,
        outline='black',
        show_conf=False,
        scale=False,
        use_font='Arial.ttf'
        ):
        """Render a page.

        :param tokens: The tokens to draw
        :type tokens: list
        :param bboxes: The bounding boxes to use
        :type bboxes: list
        :param scores: The confidence scores for each token
        :type scores: list
        :param outline: Color of the bounding boxes, pass None for no bounding box
        :type outline: str or None
        :param show_conf: Change the opacity of tokens to reflect their confidence score
        :type show_conf: bool
        :param scale: Scale the font size for each token to fill the bounding box
        :type scale: bool
        :param use_font: Font to use (can be a path)
        :type use_font: str
        :returns: Rendered page
        :rtype: PIL image
        """
        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype(use_font)

        for token, bbox, score in zip(tokens, bboxes, scores):
            draw.rectangle(bbox, outline=outline)

            if scale:
                font_size = self._scale_font(token, bbox, font_size=16, use_font='Arial.ttf')
                font = ImageFont.truetype(use_font, font_size)

            draw.text(
                xy=bbox[:2],
                text=token,
                fill='black',
                font=font,
                align='center',
                anchor='ms'
            )

        return self.img

    def _scale_font(
        self,
        token,
        bbox,
        font_size=16,
        use_font='Arial.ttf'
        ):
        """For a given token, find the font size that best fills the bounding box area.

        :param token: Text to render
        :type token: str
        :param bbox: The bounding box to fill
        :type bbox: list
        :param font_size: The size to start at
        :type font_size: int
        :param use_font: The font to use
        :type use_font: str
        :returns: Font size that best approximates the bounding box area
        :rtype: int
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        size = None
        font = ImageFont.truetype(use_font, font_size)
        while (size is None or (size[0] * size[1]) > area) and font_size > 0:
            font = ImageFont.truetype(use_font, font_size)
            size = font.getsize(token)
            font_size -= 1

        return font_size
