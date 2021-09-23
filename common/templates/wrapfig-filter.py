#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pandoc filter to allow variable wrapping of LaTeX/pdf documents
through the wrapfig package.

Simply add a " {?}" tag to the end of the caption for the figure, where
? is an integer specifying the width of the wrap in inches. 0 will 
cause the width of the figure to be used.

"""
from pandocfilters import toJSONFilter, Image, RawInline, stringify
import re, sys

FLAG_PAT = re.compile('.*\{(\d+\.?\d?)\}')

def wrapfig(key, val, fmt, meta):
    if key == 'Image':
        attrs, caption, target = val
        if FLAG_PAT.match(stringify(caption)):
            # Strip tag
            size = FLAG_PAT.match(caption[-1]['c']).group(1)
            stripped_caption = caption[:-2]
            if fmt == 'latex':
                latex_begin = r'\begin{wrapfigure}{r}{' + size + 'in}'
                if len(stripped_caption) > 0:
                    latex_fig = r'\centering\includegraphics{' + target[0] \
                                + '}\caption{'
                    latex_end = r'}\end{wrapfigure}'
                    return [RawInline(fmt, latex_begin + latex_fig)] \
                            + stripped_caption + [RawInline(fmt, latex_end)]
                else:
                    latex_fig = r'\centering\includegraphics{' + target[0] \
                                + '}'
                    latex_end = r'\end{wrapfigure}'
                    return [RawInline(fmt, latex_begin + latex_fig)] \
                            + [RawInline(fmt, latex_end)]
            else:
                return Image(attrs, stripped_caption, target)

if __name__ == '__main__':
    toJSONFilter(wrapfig)
    sys.stdout.flush() # Should fix issue #1 (pipe error)
