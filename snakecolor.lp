#const nn=6.
#const mm=6.

#program applecolor(x,y).
apple((x,y)).

attr(node, (I,J), label, "🍎") :-  apple((I,J)).
attr(node, (I,J), fontsize, 28) :-  apple((I,J)).

#program snakecolor(x,y,t).
snake((x,y),t).
snakenext(X,(x,y)) :- snake((x,y),N+1), snake(X,N).


#program pathcolor(x,y,t).
path((x,y),t).
next(S,F) :- path(S,N), path(F,N+1).
next(E,A) :- path(A,1), path(E,nn*mm).

edge((p(X),p(Y)), snake) :- next(Y,X).
attr(edge, (p(X),p(Y)), penwidth, "4") :- next(Y,X).
attr(edge, (p(X),p(Y)), color, "#555555") :- next(Y,X).
attr(edge, (p(X),p(Y)), arrowhead, "inv") :- next(Y,X).
attr(edge, (p(X),p(Y)), dir, "forward") :- next(Y,X).


#program basecolor().

field((X,Y)) :- X=1..nn, Y=1..mm.
end(Y) :- snake(Y,X), not snake(_,X+1).
endidx(N) :- end(X), snake(X,N).
head(XY):- snake(XY,1).

graph(snake).
node(XY, snake) :- field(XY).

attr(graph, default, label, "snake").
attr(node, XY, color, @colorn(N,E)):- field(XY), snake(XY,N), endidx(E). % body nodes
attr(node, XY, color, @colorn(N,E)):- field(XY), snake(XY,N), endidx(E), head(XY).
attr(node, XY, color, transparent):- field(XY), not snake(XY,_) .
attr(node, XY, shape, square):- field(XY).
attr(node, XY, style, filled):- field(XY).


node(p((X,Y)), snake) :- field((X,Y)).
attr(node, p((X,Y)), label, "") :- field((X,Y)).
attr(node, p((X,Y)), pos, @poss(X,Y)) :- field((X,Y)).
attr(node, p((X,Y)), height, "0") :- field((X,Y)).
attr(node, p((X,Y)), color, transparent) :- field((X,Y)).
attr(node, p((X,Y)), style, filled) :- field((X,Y)).
attr(node, p((X,Y)), shape, square):- field((X,Y)).


attr(node, (I,J), label, "") :-  field((I,J)).
attr(node, (I,J), label, "👀") :-  head((I,J)). %👀 🐍
attr(node, (I,J), fontsize, 28) :-  head((I,J)).
attr(node, (I,J), pos, @poss(I,J)) :- field((I,J)).
attr(node, (I,J), height, @intervalFloat(N,nn*mm,75,25)) :- snake((I,J),N). % 0.75
attr(node, (I,J), height, "0") :- field((I,J)), not snake((I,J),_).

node(s(X,Y), snake) :- field((X,Y)).
attr(node, s(X,Y), label, "") :- field((X,Y)).
attr(node, s(X,Y), pos, @poss(X,Y)) :- field((X,Y)).
attr(node, s(X,Y), height, @intervalFloat(N,nn*mm,75,25)) :- field((X,Y)), snake((X,Y),N). %0.75
attr(node, s(X,Y), height, "0.75") :- field((X,Y)), not snake((X,Y),_).
attr(node, s(X,Y), color, transparent) :- field((X,Y)).
attr(node, s(X,Y), style, filled) :- field((X,Y)).
attr(node, s(X,Y), shape, square):- field((X,Y)).

edge((s(X,Y),s(XX,YY)), snake) :- snake((X,Y),N),snake((XX,YY),N+1). 
attr(edge, (s(X,Y),s(XX,YY)), color, @colornbetween(N,E)):- snake((X,Y),N),snake((XX,YY),N+1),  endidx(E). % body edges
attr(edge, (s(X,Y),s(XX,YY)), penwidth, @interval(N,nn*mm,51,17)):- snake((X,Y),N),snake((XX,YY),N+1). %44


node(g(X,Y), snake) :- field((X,Y)), X+Y=2*N, N=1..nn+mm.
attr(node, g(X,Y), label, "") :- field((X,Y)), X+Y=2*N, N=1..nn+mm.
attr(node, g(X,Y), pos, @poss(X,Y)) :- field((X,Y)), X+Y=2*N, N=1..nn+mm.
attr(node, g(X,Y), height, "1") :- field((X,Y)), X+Y=2*N, N=1..nn+mm.
attr(node, g(X,Y), color, "#EEEEEE") :- field((X,Y)), X+Y=2*N, N=1..nn+mm.
attr(node, g(X,Y), style, filled) :- field((X,Y)), X+Y=2*N, N=1..nn+mm.
attr(node, g(X,Y), shape, square):- field((X,Y)).
