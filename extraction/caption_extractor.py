#!/usr/bin/env python3

import re

def main():
    with open('example_caption.txt', 'r') as f:
        lines = f.readlines()

    # Fit the patter of text lines, and form a group around the actual text content 
    group_out_text_regex = '<s.*>(.*)</s>'
    # list to store text content
    text_lines = []
    for line in lines:
        if line.startswith('<s '):
            text_lines.append(re.search(group_out_text_regex, line).group(1))
    print(' '.join(text_lines))

if __name__ == '__main__':
    main()
