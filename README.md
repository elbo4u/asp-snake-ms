# asp-snake-ms

Script to let the computer play snakes! in clingo! using python! and multi-shot!


How to use this script:
python snakes.py [-h] [-n N] [-m M] [-to TO] [-a {oneshot,adhoc,preground,assume,nogood,oneshot}] [-g] [-gg] [-v V] [-H]

  -n N                  dimension 1 of the grid
  -m M                  dimension 2 of the grid
  -to TO                timeout in seconds
  -a {os,redo,adhoc,preground,assume,nogood,oneshot,ms_redo,ms_preground,ms_assume,ms_nogood,o,r,p,a,n,1,2,3,4,5} approaches
  -g                    create minimal graphics
  -gg                   create extensive graphics
  -v V                  output activitiy (0,1,2)
  -H                    disable heuristics

example:   python snakes.py -n 8 -m 8 -a adhoc -to 60 -v 0 
will play on an 8x8 grid, in modus adhoc and generating graphics (into folder "out/NNNN" of currrent directory)