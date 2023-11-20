#const nn=6.
#const mm=6.


pos(1..nn*mm).

%#defined dummy/0.
#defined head/1.
%appleN(nn*mm+1):- dummy.
{dummy}.

field((X,Y)) :- X=1..nn, Y=1..mm.
connected((X,Y1),(X,Y2)) :- |Y1-Y2|=1, field((X,Y1)), field((X,Y2)).
connected((X1,Y),(X2,Y)) :- |X1-X2|=1, field((X1,Y)), field((X2,Y)).

{next(X,Y):field(X), connected(X,Y)} = 1 :- field(Y).
{next(X,Y):field(Y), connected(X,Y)} = 1 :- field(X).

path(XY,1) :- field(XY), head(XY).  % start at head
path(F,N+1) :- path(S,N), next(S,F), pos(N+1). % propagate values
:- field(F), not path(F,_).  % every path a field
:- path(F1,N), path(F2,N), not F1=F2. % not two fields same path
:- path(A,1), path(E,nn*mm), not connected(A,E). % cycle
next(E,A) :- path(A,1), path(E,nn*mm). % force cycle

#show path/2.

#minimize{N:appleN(N)}.

appleN(N) :- apple((X,Y)), path((X,Y),N).



%#heuristic next(X,Y): heur(X,Y), dummy. [ 10@2, true]
#heuristic dummy. [ 99, true]
next(X,Y) :- dummy, heur(X,Y).
%#heuristic next(X,Y): connected(X,Y). [ 1@1, level]

#external heur(X,Y) : connected(X,Y).
%#external maxdist(N) : N = 2..nn*mm.
%#heuristic next(X,Y): heur(X,Y), maxdist(N), appleN(N). [ 10, true]
%#heuristic appleN(N): maxdist(N). [ 1, sign]
%#heuristic appleN(N): maxdist(N). [ 100, init]


% -------------------------------- ONESHOT - OS --------------------------------

#program applefact(x,y).
apple((x,y)).

#program snakefact(x,y,t).
snake((x,y),t).

#program osShortcut().
#defined snake/2.
head(XY):- snake(XY,1).
:- snake(X,1), snake(Y,2), next(X,Y). %no movement towards neck allowed
end(X) :- snake(X,N), not snake(_,N+1).
endidx(N) :- end(X), snake(X,N).
:- snake(X,P), path(X,N), endidx(L), P-1+N<L+1, N>1. 

#program osNaive().
head(XY):- snake(XY,1).
next(Y,X):- snake(X,N), snake(Y,N+1).

#program markSnake().
snakeI(X,L-N+1) :- endidx(L), snake(X,N).

:- not next(X,Y), appleN(A), snakeI(X,S), S>A-2, snakeI(Y,S+1).


% -------------------------------- MULTISHOT - MS --------------------------------



#program extApples().
#external apple(XY) : field(XY).

#program extHeads().
#external head((X,Y)) : field((X,Y)).

#program extHeadsMirror().
#external head((X,Y)) : field((X,Y)), X<=Y, X*2<=nn, Y*2<=mm, mm=nn.
#external head((X,Y)) : field((X,Y)),  X*2<=nn, Y*2<=mm, not mm=nn.

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
    #external nextext(X1,Y1,X2,Y2,L) : connected((X1,Y1),(X2,Y2)), L = 0..nn*mm-2.
    %:- appleN(A), path((X1,Y1),S),  nn*mm-L-1<S-A, nextext(X1,Y1,X2,Y2,L), not next((X1,Y1),(X2,Y2)).

    %:- appleN(A), s>A-2, redo(num), not next((x1,y1),(x2,y2)).
    :- appleN(A),  A<LS, nextext(X1,Y1,X2,Y2, LS+3), not next((X1,Y1),(X2,Y2)).

    %#external smaller(X,Y,S) : field((X,Y)), S=2..(nn*mm-1).
    %:- appleN(A), S<A-3, path((X,Y),N), N < S, field((X,Y)), smaller(X,Y,S).
    

#program consPre().
    #external smaller(X,Y,S) : field((X,Y)), S=2..(nn*mm-1).
    :- path((X,Y),N), N < S, field((X,Y)), smaller(X,Y,S).





% -------------------------------- ms assume --------------------------------

#program assume(). 
    smaller(X,Y,S) :- path((X,Y),N), N < S, field((X,Y)), pos(S).


#program assumeMark().
   0 {snake(XY,S):pos(S)} 1 :- field(XY).
   :- not next((X,Y),(X1,Y1)), appleN(A), snake((X,Y),S), S>A-2, snake((X1,Y1),S+1), connected((X1,Y1),(X,Y)).




% -------------------------------- ms nogood --------------------------------

#program nogoodBase().

{dummy}.

#program ignore(t).

ignore(t).



% -------------------------------- MULTISHOT - MS --------------------------------



