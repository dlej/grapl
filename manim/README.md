# `manim` animations

This directory contains the code used to generate the video files used in the making of our AISTATS 2020 presentation using the [`manim`](https://github.com/3b1b/manim) library from mathematics YouTuber [3Blue1Brown](https://www.3blue1brown.com/).

After installing `manim`, to compile the videos, run in this directory
```
$ python -m manim aistats.py <Scene>
```
where `<Scene>` is the desired `Scene` class in [`aistats.py`](https://github.com/dlej/grapl/blob/master/manim/aistats.py)