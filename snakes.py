import sys
from clingo import Control, Number, Function, String
import timeit #timeout
from datetime import datetime
import random #apple
from clingraph.graphviz import render # draw 
from clingraph.orm import Factbase # draw image
from clingraph.graphviz import compute_graphs
import matplotlib  # color snake
from PIL import Image as PILImage
import os
import itertools
import copy
import argparse


def list2Number(xy):
    l = []
    #print(xy)
    for x in xy:
        l.append(Number(x))
    return l

def apple2symbol(apple, predicate = "applefact"):  # for grounding
    return (predicate,[Number(apple[0]),Number(apple[1])])

def snake2symbol(snake, predicate = "snakefact"): # for grounding
    toground =[]
    for i,xy in enumerate(snake):
        toground.append((predicate,[Number(xy[0]),Number(xy[1]),Number(i+1)]))
    return toground


def rotatepath(path):
    path.append(path[0]) # rotate once
    return path[1:]


def pairwise(iterable):    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def next2path(head,next): # convert list of next tuples into path
    path=[]
    cur =tuple(head)
    while len(path)<len(next):
        path.append(list(cur))
        cur = next[cur]
    return path


strat = {
			"conservative":1,
			#"naive":1,
			"shortcut":2,
			"hybrid":3,
			"c":1,
			#"n":1,
			"s":2,
			"h":3,
			"1":1,
			"2":2,
			"3":3,}

def easyhc(n, m): # generate easy Hamiltonian cycle as start
    result = []
    
    flip = False
    for j in range(1,m+1):
        tmp = [(j, i+1) for i in range(1,n)]
        if flip:
            tmp.reverse()
        flip = not flip
        result.extend(tmp)
    result.extend([( i,1) for i in range(n ,0,-1)])
    return result

class SnakeContext: # for grounding involving python
    def poss(self, x,y, scale=1):
        scale = float(str(scale).strip('"'))
        x = int(float(str(x))*scale)
        y = int(float(str(y))*scale)
        return String(f"{x},{y}!")
    
    def interval(self, i, maxim, upper, lower):
        y = 1.0- float(int(str(i))) / float(int(str(maxim)))
        y = (float(int(str(upper))-int(str(lower))))*y+int(str(lower))
        return String(str(int(y)))
    def intervalFloat(self, i, maxim, upper, lower):
        y = 1.0- float(str(i)) / float(str(maxim))
        diff = int(str(upper))-int(str(lower))
        y = y*diff+int(str(lower))
        return String(str(float(y/100)) )
    
    def colori(self, i):
        i = int(str(i))
        (r, g, b) = (48+i*5,  i*10 , 150 + 10*i)
        return String("#{:02x}{:02x}{:02x}".format(r, g, b))
    
    def colorn(self, i, n, a="AA"):
        a= str(a)
        i = int(str(i))
        n = int(str(n))
        n = max(5,n)
        x = (1.0-float(i-1)/float(n-1))*0.5+0.3
        x=x*0.4+0.25
        rgb =  matplotlib.colormaps['viridis'](x)
        s = '#%02x%02x%02x'+ a
        return String(s % tuple(int(255*x) for x in rgb[:3]))
    def colornbetween(self, i, n, a="AA"):
        a= str(a)
        i = int(str(i))
        n = int(str(n))
        n = max(5,n)
        x = (1.0-float((float(i)+0.5)-1)/float(n-1)) *0.5+0.3
        x=x*0.4+0.25
        rgb =  matplotlib.colormaps['viridis'](x)
        s = '#%02x%02x%02x'+ a
        return String(s % tuple(int(255*x) for x in rgb[:3]))
    
    def sizeIN(self, i, n, oben="0.8", unten="0.2"):
        i = int(str(i))
        n = int(str(n))
        oben = float(str(oben))
        unten = float(str(unten))
        n = max(5,n)
        x = (1.0-float(i-1)/float(n-1))*(oben-unten) +unten
        return String(str(x))
    
    def sizeINbetween(self, i, n, a="AA"):
        i = int(str(i))
        n = int(str(n))
        n = max(5,n)
        x = (1.0-float((float(i)+0.5)-1)/float(n-1))*0.6*0.8+0.2
        return String(str(x))
        

class snakeviz: # class to handle graphics
    def find_lowest_non_folder_number(self, folder_path):
        for i in range(100000):
            number_str = str(i).zfill(5)  # Zero-pad the number to 5 digits
            file_path = os.path.join(folder_path, number_str)
            if not os.path.isdir(file_path):
                return number_str  # Return the first number that is not a folder
        return -1  # Return -1 if all numbers are folders

    def __init__(self, n, m, grafik):
        self.n = n
        self.m = m
        self.cnt = 0
        self.iter = 0
        self.seed = 99999#random.randint(10000,99999)
        self.images = []
        self.current = None
        self.grafik = grafik
        self.folder = self.find_lowest_non_folder_number("./out/")
        #print("image folder ", self.seed)
    
    def nextiter(self):
        self.iter = self.iter+1

    def getimages(self):
        return [x for x in self.images+[self.current] if x is not None]

    def gengif(self):
        if  self.grafik==0:
            return
        images = []
        for filename in self.images:
            images.append(PILImage.open("out/"+filename))
        images[0].save("out/"+str(self.folder)+'animation.gif', save_all=True, append_images=images[1:], duration=len(self.images)/4, loop=0)
        #Image(filename="out/"+str(self.seed)+'animation.gif')


    def printsnake(self, sn, app, pa, rotate=0, cost=0):
        if  self.grafik==0:
            return
        snake = copy.deepcopy(sn)
        path = copy.deepcopy(pa)
        apple = copy.deepcopy(app)
        
        if rotate!=0:
            if rotate[0]:
                for i in range(len(snake)):
                    snake[i][0] = self.n+1-snake[i][0]
                for i in range(len(path)):
                    path[i][0] = self.n+1-path[i][0]
                if apple!=None:
                    apple[0] = self.n+1-apple[0]

            if rotate[1]:
                for i in range(len(snake)):
                    snake[i][1] = self.m+1-snake[i][1]
                for i in range(len(path)):
                    path[i][1] = self.m+1-path[i][1]                
                if apple!=None:
                    apple[1] = self.m+1-apple[1]

            if rotate[2]:
                for i in range(len(snake)):
                    snake[i][0], snake[i][1] = snake[i][1], snake[i][0]
                for i in range(len(path)):
                    path[i][0], path[i][1] = path[i][1], path[i][0]
                if apple!=None:
                    apple[0], apple[1] = apple[1], apple[0]

        tmp = Control(["-c", "nn="+str(self.n),"-c", "mm="+str(self.m)])
        tmp.load("snakecolor.lp")
        toground = [("basecolor",[]) ]  
        if apple is not None:
            toground.append(apple2symbol(apple,"applecolor"))
        toground.extend(snake2symbol(snake,"snakecolor"))
        toground.extend(snake2symbol(path,"pathcolor"))
        #print(toground)

        tmp.ground(toground, context=SnakeContext())
        fbs = []
        tmp.solve(on_model=lambda m: fbs.append(Factbase.from_model(m)))
        graphs = compute_graphs(fbs)
        mystr= ""
        if cost > 0:
            mystr = +"_"+str(cost)
        #filename = str(self.seed)+"/snake"+"{:05d}".format(self.cnt)+"_"+"{:02d}".format(self.iter)+mystr
        filename = str(self.folder)+"/snake"+"{:05d}".format(self.cnt)+mystr
        self.cnt += 1

        if cost > 0:
            self.current = filename+".png"
        else:
            if self.current is not None:
                self.images.append(self.current)
                self.current = None
            self.images.append(filename+".png")

            
        render(graphs,name_format=filename, format='png', engine = "neato")
        return filename+".png"







class snakeC:  # snake class

    def print(self, level=2, message="", *args, **kwargs):
        if self.verbosity >= level:
            print(message, *args, **kwargs)

    def __init__(self,args):
        self.n = args.n
        self.m = args.m
        if self.m == 0:
            self.m = self.n
        self.to = args.to # timeout
        self.grafik = 0 # draw images?
        if args.gg:
            self.grafik = 1 # draw images?
        if args.g:
            self.grafik = 2 # draw images?
        self.strategy = "conservative"
        self.rotate = args.x
        if self.rotate not in [0, False]:
            self.rotate = [0,0,0]
            self.rotateOld = [0,0,0]
        self.useheuristics = args.H
        self.verbosity = args.v

            
        self.allfields = [[i, j] for i in range(1, self.n+1) for j in range(1, self.m+1)]
        self.snakevis = snakeviz(self.n, self.m, self.grafik)
        self.path = []
        self.next = {}
        self.snake = [[1,1]]
        self.apple = []
        self.timesolve = 0.0
        self.timegroundinit = 0.0
        self.timeground = 0.0
        self.timegroundList = []
        self.timesolveList = []
        self.steps = 0
        self.appleSteps = 0
        self.stepList = []
        self.mdList = []
        self.optimizeList = []
        self.md = 0
        self.optimal = 0
        self.cost = 0
        self.solution = 0
        self.unsat = []
        self.skip = False
        self.encoding = "snake.lp"
        self.assume = []
        self.boobytrap = 0
        self.tmpoptimize = []
        self.mixup = False
        self.mixupResolved = ""
        #self.seed = random.randint(0,999999)
        self.important = []
        self.heurlist = []
        self.notgen = False
        self.timeout = False 
        self.ctllist = ["-c", "nn="+str(self.n),"-c", "mm="+str(self.m), "0", "--heuristic=Domain"]
        

        if not hasattr(self, 'name'):
            self.name = ""

        if len(self.initground) > 0 :
            self.ctl = Control(self.ctllist)
            self.ctl.load(self.encoding) 
            #if strat[self.strategy] == strat["hybrid"]:
            #    self.initground.append(("extMark",[]))
            self.tic = timeit.default_timer()
            self.ctl.ground(self.initground  )
            self.toc = timeit.default_timer()
            self.timegroundinit += (self.toc - self.tic)
            self.timeground += (self.toc - self.tic)
            self.print(1,"ground", 0, self.timeground, (self.toc - self.tic))
            self.print(2,"initground", self.initground)

        self.print(0,"snake_statistics", self.name, self.strategy, datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        self.print(0,"n", self.n)
        self.print(0,"m", self.m)
        self.print(0,"to", self.to)
        self.print(0,"grafik", self.grafik)
        self.print(0,"encoding", self.encoding)
        self.print(0,"rotate", self.rotate)
        self.print(0,"verbosity", self.verbosity, flush=True)

   

    def genapple(self):
        applefields = [xy for xy in self.allfields if xy not in self.snake]
        if len(applefields)==0:
            self.apple = None
        else:
            self.apple = random.choice(applefields)
            diff = abs(self.apple[0]-self.snake[0][0]) + abs(self.apple[1]-self.snake[0][1])
            self.print(1,"appleMD", len(self.snake), diff, self.apple, self.snake[0])
            self.md += diff
            self.mdList.append(diff)
    
    
    #def next2path(head,next):

    def mymodel(self,model):
        s = str(model.cost)
        self.cost = int(s[1:-1])-1
        self.tmpoptimize.append(self.cost)
        #self.print(model.number, model.symbols(shown=True))
        if model.contains(Function("dummy",[])):
            self.print(2,"dummy", len(self.snake))
        self.print(0,"solution", len(self.snake), self.cost, (timeit.default_timer() - self.tic)*1000, flush=True)
        self.path = [[0,0]]*(self.n*self.m)

        self.next={}
        for a in model.symbols(shown=True):
            if a.name == "next" and len(a.arguments) == 2:
                xy1 = a.arguments[0]
                xy2 = a.arguments[1]
                x1 = xy1.arguments[0].number
                y1 = xy1.arguments[1].number
                x2 = xy2.arguments[0].number
                y2 = xy2.arguments[1].number
                self.next[(x1,y1)]=(x2,y2)
        if len(self.next)>0:
            self.path = next2path(self.snake[0],self.next)
        self.path = rotatepath(self.path)
        self.solution = len(self.snake)

    def mirror(self, l):
        if self.rotate!=0:
            if self.rotate[0]:
                for i in range(len(l)):
                    l[i][0] = self.n+1-l[i][0]
            if self.rotate[1]:
                for i in range(len(l)):
                    l[i][1] = self.m+1-l[i][1]
            if self.rotate[1]:
                for i in range(len(l)):
                    l[i][0], l[i][1] = l[i][1], l[i][0]
        return l

    def eatapple(self):
        if self.path[0] == self.snake[0]:
            self.path = rotatepath(self.path)
            self.print(0,"No_new_path_generated", len(self.snake))
            self.important.append(len(self.snake))
            if self.mixup and self.solution<len(self.snake): #mixup issue
                self.mixup = False
            else:
                self.notgen = True
                return

        init=True    
        self.appleSteps += 1
        while len(self.path) > 0 and self.apple != self.path[0] and self.apple is not None and len(self.snake)<self.n*self.m:
            if self.grafik == 1 or init:
                self.snakevis.printsnake(self.snake, self.apple, self.path, self.rotate)
                init = False
            if self.apple not in self.path:
                break
            if self.path[0] in self.snake[:-1]:
                    self.boobytrap = len(self.snake)
                    self.print(0,"SnakeBitItself", self.path[0], self.snake[:-1], flush=True)
                    break
            self.snake = [self.path[0]]+self.snake[:-1]
            self.path = rotatepath(self.path)
            self.appleSteps += 1

        self.snakevis.printsnake(self.snake, self.apple, self.path, self.rotate)
        #self.snakevis.nextiter()
        if len(self.path)>0 and not self.skip:
            self.snake = [self.path[0]]+self.snake
        self.skip = False

        #--------------mirror------------
        if self.rotate!=0:
            snake =  copy.deepcopy(self.snake)
            path =  copy.deepcopy(self.path)
            rotate = copy.deepcopy(self.rotate)
            if snake[0][0] > (self.n >> 1):
                rotate[0] = 1-rotate[0]
                for i in range(len(snake)):
                    snake[i][0] = self.n+1-snake[i][0]
                for i in range(len(path)):
                    path[i][0] = self.n+1-path[i][0]

            if snake[0][1] > (self.m >> 1):
                rotate[1] = 1-rotate[1]
                for i in range(len(snake)):
                    snake[i][1] = self.m+1-snake[i][1]
                for i in range(len(path)):
                    path[i][1] = self.m+1-path[i][1]

            if self.n==self.m and snake[0][0] > snake[0][1]:
                rotate[2] = 1-rotate[2]
                for i in range(len(snake)):
                    snake[i][0], snake[i][1] = snake[i][1], snake[i][0]
                for i in range(len(path)):
                    path[i][0], path[i][1] = path[i][1], path[i][0]
            self.snake =  copy.deepcopy(snake)
            self.path =  copy.deepcopy(path)
            self.rotateOld =  copy.deepcopy(self.rotate)
            self.rotate =  copy.deepcopy(rotate)

            


    def notend(self):
        return len(self.snake) <= self.n*self.m and \
                self.apple is not None and \
                self.boobytrap == 0 and \
                self.notgen == False and \
                self.unsat == [] 
    

    def assignListDoubleBrackets(self, predicate, xy1, xy2, val=True):
        self.ctl.assign_external(Function(predicate,  [Function("",list2Number(xy1)), Function("",list2Number(xy2))]), val) #assign once

    def assignSnakeHeuristic(self, val):
        if self.useheuristics:
            if val == True:
                if len(self.snake)>1:
                    path = self.path
                else:
                    path = easyhc(self.n, self.m)                    
                    self.print(2,"easypath",path)
                for (x,y) in pairwise(path+[path[0]]):
                        self.assignListDoubleBrackets("heur", x, y, val)
                        self.heurlist.append(( x, y))
            if val == False:
                for ( x, y) in self.heurlist:
                    self.assignListDoubleBrackets("heur", x, y, val)
                self.heurlist = []



    def assignListBrackets(self, predicate, xy, val=True):
        self.ctl.assign_external(Function(predicate,  [Function("",list2Number(xy))]), val) #assign once

    def assignList(self, predicate, xy, val=True):
        self.ctl.assign_external(Function(predicate, list2Number(xy)), val) #assign once

    def assignLists(self, predicate, xy, val=True):
        for x in xy:
            self.assignList(predicate, x, val)

    def presolve(self):
        self.assignListBrackets("apple", self.apple, True)
        self.assignListBrackets("head", self.snake[0], True)
        self.assignSnakeHeuristic( True)
        self.tic = timeit.default_timer()
        self.tmpoptimize = []

    def handlesolve(self):
        with self.ctl.solve(assumptions = self.assume, async_=True, on_model = self.mymodel) as handle:
            while not handle.wait(1): # check every sec if finished
                 if timeit.default_timer() - self.tic > self.to and self.to>0.0: # if timeout reached:
                    handle.cancel()
                    self.timeout=True
                    self.print(0,"timeout", len(self.snake), self.cost, flush=True)
                    break  
            if handle.get().exhausted and not handle.get().unsatisfiable:
                self.print(1,"optimum", len(self.snake), self.cost, (timeit.default_timer() - self.tic)*1000, flush=True)
                if self.optimal == len(self.snake)-1:
                    self.optimal += 1
            if handle.get().unsatisfiable == True and len(self.snake)+1 < self.n*self.m:
                self.print(0,"UNSAT",len(self.snake),self.solution)
                self.unsat.append((len(self.snake),self.apple,self.snake,self.path))
                self.skip = True


    def postsolve(self):
        self.print(1,"apple", self.apple, self.n, self.m, self.cost)
        self.print(1,"snake", self.snake)
        self.print(1,"path", self.path, flush=True)
        self.assignListBrackets("apple", self.apple, False)
        self.assignListBrackets("head", self.snake[0], False)
        self.assignSnakeHeuristic(False)
        self.ctl.cleanup

        self.toc = timeit.default_timer()
        self.timesolve += (self.toc - self.tic)
        self.steps += self.cost
        self.stepList.append(self.cost)
        self.timegroundList.append(self.timeground)
        self.timesolveList.append(self.timesolve)
        self.optimizeList.append(self.tmpoptimize)
        if self.solution < len(self.snake): # valid because the snake did not change yet, only the path
            self.print(1, "notfound",len(self.snake),self.solution)

        self.print(1, "finish_search",len(self.snake),self.cost,(self.toc - self.tic)*1000, flush=True)


    def searchpath(self):
        self.presolve()
        self.handlesolve()
        self.postsolve()
        return len(self.path)>0
    

    
class ms_nogood(snakeC):
    def __init__(self,args):
        self.name = "ms_nogood"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]),  ("base",[]), ("heuristics",[]) ] #("redoBase",[])
        if args.x in [0,False]:
            self.initground.append(("extHeads",[])) 
        super().__init__( args)



    def setnexts(self, model, snake, predicate = "next", val=True): # conservative
        for i in range(len(snake) - 1):
            tmp =  list2Number(snake[i+1] + snake[i])   # reverse order 
            tmp1 = Function("",[tmp[0],tmp[1]])
            tmp2 = Function("",[tmp[2],tmp[3]])
            model.context.add_clause( [(Function(predicate, [tmp1,tmp2]), val)])
            #self.print("CHECK",len(snake),str([(Function(predicate, [tmp1,tmp2]), val)]))

    def mymodel(self,model):
        self.modelnr = model.number
        if model.number == 1: #model.contains(Function("dummy",[])):
            if not model.contains(Function("dummy",[])):
                self.print(1,"MIXUP dummy > 1",model.number, model.cost, len(self.snake))
                self.mixup = True
                self.mixupResolved += "x"
            self.setnexts( model, self.snake, "next", True)
        else:
            if model.contains(Function("dummy",[])):
                self.print(1,"MIXUP 1, not dummy",model.number,model.cost,  len(self.snake))
        super().mymodel( model)

class ms_assume(snakeC):
    def __init__(self,args):
        self.name = "ms_assume"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]), ("base",[]), ("heuristics",[]) ]
        
        if args.x in [0,False]:
            self.initground.append(("extHeads",[])) 
        super().__init__( args)

    
    def assumeNext(self, snake, predicate = "next", val=True):
        assumption = []
        for i in range(len(snake) - 1):
            tmp =  list2Number(snake[i+1] + snake[i])   # reverse order 
            tmp1 = Function("",[tmp[0],tmp[1]])
            tmp2 = Function("",[tmp[2],tmp[3]])
            assumption.append((Function(predicate, [tmp1,tmp2]), val))
        return assumption

    def presolve(self):
        self.assume = self.assumeNext(self.snake, "next", True)
        super().presolve()

class ms_preground(snakeC):
    def __init__(self,args):
        self.name = "ms_preground"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]), ("base",[]), ("heuristics",[]) ] 
        
        if args.x in [0,False]:
            self.initground.append(("extHeads",[])) 
        self.initground.append(("consNaive",[]))
        super().__init__( args)

            
    def assignNextExts(self, snake, predicate = "nextext", val=True):
        for i in range(len(snake) - 1):
            tmp =  list2Number(snake[i+1] + snake[i])   # reverse order 
            tmp1 = Function("",[tmp[0],tmp[1]])
            tmp2 = Function("",[tmp[2],tmp[3]])
            self.ctl.assign_external(Function(predicate, [tmp1,tmp2]), val)

        

    def presolve(self):
        self.assignNextExts(self.snake, "prenext", True)
        super().presolve()
        
    def postsolve(self):
        self.assignNextExts(self.snake, "prenext", False)
        super().postsolve()
        
class ms_redo(snakeC):
    def __init__(self,args):
        self.name = "ms_redo"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]), ("base",[]), ("heuristics",[]) ] 
        
        if args.x in [0,False]:
            self.initground.append(("extHeads",[])) 
        super().__init__( args)
                         
    def snake2next(self, snake, predicate = "consRedo"):
        toground =[]
        l = len(snake)
        for i in range(len(snake) - 1):
            tmp = snake[i+1] + snake[i] + [l] #change order!
            toground.append((predicate,list2Number(tmp))) 
        return toground

    def presolve(self):
        toground = [("extRedo",[Number(len(self.snake))])]
        toground.extend(self.snake2next(self.snake, "redoNaive"))
        self.tic = timeit.default_timer()
        self.ctl.ground(toground)
        self.toc = timeit.default_timer()
        self.timeground += (self.toc - self.tic)

        self.print(2,"ground", len(self.snake), self.timeground, (self.toc - self.tic))
        self.assignList("redo", [len(self.snake)], True)
        super().presolve()
        
    def postsolve(self):
        self.ctl.release_external(Function("redo", [Number(len(self.snake))]))
        super().postsolve()

class oneshot(snakeC):
    def __init__(self,args):
        self.name = "oneshot"
        self.initground = []  
        super().__init__( args)


    def presolve(self):
        self.ctl = Control(self.ctllist)
        self.ctl.load(self.encoding) 
        
        toground = [("base",[]), ("heuristics",[])  ]  
        toground.append(apple2symbol(self.apple))
        toground.extend(snake2symbol(self.snake))        
        toground.append(("osNaive",[]))

        self.tic = timeit.default_timer()
        self.ctl.ground(toground)
        self.assignSnakeHeuristic( True)
        self.toc = timeit.default_timer()
        self.timeground += (self.toc - self.tic)
        self.print(2,"ground", len(self.snake), self.timeground, (self.toc - self.tic))
        self.tic = timeit.default_timer()
             


def main():
    tstart = timeit.default_timer()

    classmap = {
			'os':oneshot,
			'redo':ms_redo,
			'adhoc':ms_redo,
			'preground':ms_preground,
			'assume':ms_assume,
			'nogood':ms_nogood,
			'oneshot':oneshot,
			'ms_redo':ms_redo,
			'ms_preground':ms_preground,
			'ms_assume':ms_assume,
			'ms_nogood':ms_nogood,
			'o':oneshot,
			'r':ms_redo,
			'p':ms_preground,
			'a':ms_assume,
			'n':ms_nogood,
			'1':oneshot,
			'2':ms_redo,
			'3':ms_preground,
			'4':ms_assume,
			'5':ms_nogood
			}

    parser = argparse.ArgumentParser(description='Script to let the computer play snakes! in clingo! using python! and multi-shot!')
    parser.add_argument('-n', help='dimension 1 of the grid', type=int, default=6)
    parser.add_argument('-m', help='dimension 2 of the grid', type=int, default=0)
    parser.add_argument('-to',  help='timeout in seconds', type=int, default=10)
    parser.add_argument('-a', help='approaches', default='os', choices=['os','redo','adhoc','preground','assume','nogood','oneshot','ms_redo','ms_preground','ms_assume','ms_nogood','o','r','p','a','n','1','2','3','4','5'],)
    parser.add_argument('-g', help='create minimal graphics', action='store_true', default=False)
    parser.add_argument('-gg', help='create extensive graphics', action='store_true', default=False)
    parser.add_argument('-x', action='store_false', default=True, help=argparse.SUPPRESS)
    parser.add_argument('-v', help='output activitiy (0,1,2)', type=int, default=2)
    parser.add_argument('-H', help='disable heuristics', action='store_false', default=True) 

    args =  parser.parse_args()
    
    mysnake = classmap[args.a](args)
    
    while mysnake.notend():
        mysnake.snakevis.printsnake(mysnake.snake,None,mysnake.path, mysnake.rotate)
        mysnake.genapple()
        if mysnake.grafik==1:
            mysnake.snakevis.printsnake(mysnake.snake,mysnake.apple, [], mysnake.rotate)
        if mysnake.apple is None:
            break
        if not mysnake.searchpath(): # solve wrapper
            print("abort", len(mysnake.snake))
            break
        mysnake.eatapple() 


    mysnake.print(2,"premature",  len(mysnake.unsat), mysnake.unsat)
    print("nmto", mysnake.n, mysnake.m, mysnake.to)
    print("steplist",mysnake.stepList)
    mysnake.print(2,"mdList",mysnake.mdList)
    mysnake.print(2,"optimizeList",mysnake.optimizeList)
    mysnake.print(2,"timegroundList",mysnake.timegroundList)
    mysnake.print(2,"timesolveList",mysnake.timesolveList)
    mysnake.print(2,"steps", mysnake.steps, mysnake.md, float( mysnake.steps)/len(mysnake.snake), float( mysnake.md)/len(mysnake.snake), mysnake.m*mysnake.n/2)
    print("applesteps", mysnake.appleSteps)
    print("ttotal", (timeit.default_timer() - tstart))
    print("tground", mysnake.timeground, mysnake.timegroundinit)
    print("tsolve", mysnake.timesolve)
    print("modus", mysnake.name, mysnake.strategy)
    print("finished", mysnake.n*mysnake.m, len(mysnake.snake), mysnake.optimal+1)
    if len(mysnake.mixupResolved) > 0:
        print("mixupResolved", mysnake.mixupResolved)
    if len(mysnake.important)>0:
        print("important",mysnake.important )
    mysnake.snakevis.gengif()

if __name__ == "__main__":
    main()
