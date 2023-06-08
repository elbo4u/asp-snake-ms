# asp-snake-ms

Script to let the computer play snakes! in clingo! using python! and multi-shot!

How to use this script:
python snakes.py M N A S G T
M, N: dimensions of grid, default 6 and 6
A: approach, default redo;    os, redo, preground, assume, nogood, oneshot, ms_redo, ms_preground, ms_assume, ms_nogood, o, r, p, a, n, 1, 2, 3, 4, 5
S: strategy, default conservative;    conservative, shortcut, hybrid, c, s, h, 1, 2, 3
G: draw grafics, 0 for no (default), 1 for yes    
T: timeout in seconds, default 10.0 
example:   python snakes.py 8 8 redo hybrid 1
will play on an 8x8 grid, in modus redo using strategy hybrid and generating graphics (into folder out)