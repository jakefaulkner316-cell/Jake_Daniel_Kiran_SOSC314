"""Finish docs/index.html after nbconvert: set the page title and add alt text to the figures.

Run by build_html.sh. Takes the path of the HTML file as its only argument.
"""
import re
import sys
from pathlib import Path

TITLE = "Advertising Sentiment and the Business Cycle: Evidence from The Economist Advertisements"
ALT = [
    "Two line charts, 1948 to 2014, of the quarterly share of positive words under Loughran-McDonald and Harvard, "
    "with recession quarters shaded and decade averages. Both fall to a low around 1970 and rise afterwards; "
    "the decade-to-decade drift is far larger than the recession gap.",
    "Three bar charts comparing filter versions v1, v2 and v3 on the held-out labels: non-ads caught rises from "
    "5% to 35% to 65%, Cohen's kappa from 0.09 to 0.47 to 0.72, and the share of the corpus removed from 1.9% "
    "to 2.2% to 7.2%.",
    "Four panels of point estimates with 95% intervals for the recession effect on each sentiment outcome along "
    "five model steps, unfiltered versus the cleanest filter. Most intervals cross zero; filtering barely moves "
    "the estimates.",
    "Point estimates with 95% intervals for the change in ten topic shares in recession quarters. Technology and "
    "information falls and industrial equipment rises; the other eight topics are not significant after Holm correction.",
]

path = Path(sys.argv[1])
html = path.read_text(encoding="utf-8")
html = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", html, count=1, flags=re.S)
PLACEHOLDER = 'alt="No description has been provided for this image"'
n = html.count(PLACEHOLDER)
if n != len(ALT):
    sys.exit(f"Expected {len(ALT)} figures without alt text, found {n}. Update ALT in postprocess_html.py.")
for alt in ALT:
    html = html.replace(PLACEHOLDER, f'alt="{alt}"', 1)
path.write_text(html, encoding="utf-8")
print(f"{path}: title set, alt text added to {len(ALT)} figures")
