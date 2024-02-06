#const nn=6.
#const mm=6.

pos(1..nn*mm).

#defined head/1.
{dummy}.

field((X,Y)) :- X=1..nn, Y=1..mm.
connected((X,Y1),(X,Y2)) :- |Y1-Y2|=1, field((X,Y1)), field((X,Y2)).
connected((X1,Y),(X2,Y)) :- |X1-X2|=1, field((X1,Y)), field((X2,Y)).

{next(X,Y):field(X), connected(X,Y)} = 1 :- field(Y).
{next(X,Y):field(Y), connected(X,Y)} = 1 :- field(X).

path(XY) :- field(XY), head(XY).  % start at head
path(F) :- path(S), next(S,F). % propagate values
:- field(F), not path(F).  % every path a field

apath(XY) :- field(XY), head(XY).  % start at head
apath(F) :- apath(S), next(S,F), not apple(S). % propagate values

#show next/2.

#minimize{1,XY:apath(XY)}.


#program heuristics(). % heuristic to infuse default path
    #heuristic dummy. [ 99, true]
    next(X,Y) :- dummy, heur(X,Y).
    #external heur(X,Y) : connected(X,Y).

% -------------------------------- ONESHOT - OS --------------------------------

#program applefact(x,y).
    apple((x,y)).

#program snakefact(x,y,t).
    snake((x,y),t).

#program osNaive().
    head(XY):- snake(XY,1).
    next(Y,X):- snake(X,N), snake(Y,N+1).



% -------------------------------- MULTISHOT - MS --------------------------------

#program extApples().
    #external apple(XY) : field(XY).

#program extHeads().
    #external head((X,Y)) : field((X,Y)).

#program extHeadsMirror(). % use symmetry of the field
    #external head((X,Y)) : field((X,Y)), X<=Y, X*2<=nn, Y*2<=mm, mm=nn.
    #external head((X,Y)) : field((X,Y)), X*2<=nn, Y*2<=mm, not mm=nn.


% -------------------------------- ms redo --------------------------------


#program extRedo(num). 
    #external redo(num).


#program redoNaive(x1,y1,x2,y2, num).
    :- redo(num), not next((x1,y1),(x2,y2)). 


% -------------------------------- ms preground --------------------------------

#program consNaive().
    #external prenext(X,Y) : connected(X,Y).
    :- prenext(X,Y), not next(X,Y), connected(X,Y).





