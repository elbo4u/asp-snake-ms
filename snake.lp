#const nn=6.
#const mm=6.


pos(1..nn*mm).

#defined dummy/0.
appleN(nn*mm+1):- dummy.

field((X,Y)) :- X=1..nn, Y=1..mm.
connected((X,Y1),(X,Y2)) :- |Y1-Y2|=1, field((X,Y1)), field((X,Y2)).
connected((X1,Y),(X2,Y)) :- |X1-X2|=1, field((X1,Y)), field((X2,Y)).
{next(X,Y):field(X), connected(X,Y)} = 1 :- field(Y), not dummy.
{next(X,Y):field(Y), connected(X,Y)} = 1 :- field(X), not dummy.


path((X,Y),1) :- field((X,Y)), head(X,Y), not dummy.  % start at head
path(F,N+1) :- path(S,N), next(S,F), pos(N+1), not dummy. % propagate values
:- field(F), not path(F,_), not dummy.  % every path a field
:- path(F1,N), path(F2,N), not F1=F2, not dummy. % not two fields same path
:- path(A,1), path(E,nn*mm), not connected(A,E), not dummy. % cycle
next(E,A) :- path(A,1), path(E,nn*mm), not dummy. % force cycle

#show path/2.

#minimize{N:appleN(N)}.

appleN(N) :- apple(X,Y), path((X,Y),N), not dummy.


% -------------------------------- ONESHOT - OS --------------------------------

#program applefact(x,y).
apple(x,y).

#program snakefact(x,y,t).
snake((x,y),t).

#program osShortcut().
#defined snake/2.
head(X,Y):- snake((X,Y),1).
:- snake(X,1), snake(Y,2), next(X,Y). %no movement towards neck allowed
end(X) :- snake(X,N), not snake(_,N+1).
endidx(N) :- end(X), snake(X,N).
:- snake(X,P), path(X,N), endidx(L), P-1+N<L+1, N>1. 

#program osNaive().
head(X,Y):- snake((X,Y),1).
next(Y,X):- snake(X,N), snake(Y,N+1).

#program markSnake().
snakeI(X,L-N+1) :- endidx(L), snake(X,N).

:- not next(X,Y), appleN(A), snakeI(X,S), S>A-2, snakeI(Y,S+1).


% -------------------------------- MULTISHOT - MS --------------------------------



#program extApples().
#external apple(X,Y) : field((X,Y)).

#program extHeads().
#external head(X,Y) : field((X,Y)).

% -------------------------------- ms redo --------------------------------


#program extRedo(num). 
    #external redo(num).

#program consRedo(x,y,s,num).
    :- redo(num), path((x,y),N), N < s.

#program redoNaive(x1,y1,x2,y2, num).
    :- redo(num), not next((x1,y1),(x2,y2)). 


#program redoHybrid(x1,y1,x2,y2,s,num).
    :- not next((x1,y1),(x2,y2)), appleN(A), s>A-2, redo(num).

% -------------------------------- ms preground --------------------------------

#program consNaive().
    #external nextext(X1,Y1,X2,Y2) : connected((X1,Y1),(X2,Y2)).
    :- nextext(X1,Y1,X2,Y2), not next((X1,Y1),(X2,Y2)).

#program consHybrid().
    #external nextext(X1,Y1,X2,Y2) : connected((X1,Y1),(X2,Y2)).
    :- appleN(A), path((X1,Y1),S),  S>A-2, nextext(X1,Y1,X2,Y2), not next((X1,Y1),(X2,Y2)).


#program consPre().
    #external smaller(X,Y,S) : field((X,Y)), S=2..(nn*mm-1).
    :- path((X,Y),N), N < S, field((X,Y)), smaller(X,Y,S).





% -------------------------------- ms assume --------------------------------

#program assume(). 
    smaller(X,Y,S) :- path((X,Y),N), N < S, field((X,Y)), pos(S).


#program assumeMark().
   0 {snake(X,Y,S):pos(S)} 1 :- field((X,Y)).
   :- not next((X,Y),(X1,Y1)), appleN(A), snake(X,Y,S), S>A-2, snake(X1,Y1,S+1), connected((X1,Y1),(X,Y)).




% -------------------------------- ms nogood --------------------------------

#program nogoodBase().

{dummy}.



% -------------------------------- MULTISHOT - MS --------------------------------



