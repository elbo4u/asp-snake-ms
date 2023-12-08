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


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

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
    def largereqString(self, y):
        y = int(str(y))
        return String(f"≥{y}")
    def largerString(self, y):
        y = int(str(y))
        return String(f">{y}")
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
        rgb =  matplotlib.colormaps['viridis'](x)
        s = '#%02x%02x%02x'+ a
        return String(s % tuple(int(255*x) for x in rgb[:3]))
    def colornbetween(self, i, n, a="AA"):
        a= str(a)
        i = int(str(i))
        n = int(str(n))
        n = max(5,n)
        x = (1.0-float((float(i)+0.5)-1)/float(n-1)) *0.5+0.3
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
    def __init__(self,n,m,to,grafik,strategy, rotate):
        self.n = n
        self.m = m
        self.to = to # timeout
        self.grafik = grafik # draw images?
        self.allfields = [[i, j] for i in range(1, self.n+1) for j in range(1, self.m+1)]
        self.snakevis = snakeviz(self.n, self.m, self.grafik)
        self.path = []
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
        self.strategy = strategy
        self.rotate = rotate
        if rotate != 0:
            self.rotate = [0,0,0]
        self.assume = []
        self.boobytrap = 0
        self.tmpoptimize = []
        self.mixup = False
        self.mixupResolved = ""
        self.seed = random.randint(0,999999)
        self.important = []
        self.heurlist = []
        self.notgen = False
        self.timeout = False 
        self.ctllist = ["-c", "nn="+str(n),"-c", "mm="+str(m), "0", "--seed",str(self.seed) ,"--rand-freq",str(.01), "--heuristic=Domain"]

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
            print("ground", 0, self.timeground, (self.toc - self.tic))
            print("initground", self.initground)

        print("snake_statistics", self.name, self.strategy, datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print("n", self.n)
        print("m", self.m)
        print("to", self.to)
        print("grafik", self.grafik)
        print("encoding", self.encoding)
        print("seed", self.seed, flush=True)




        # for x in self.ctl.symbolic_atoms:
        #     if x.is_external:
        #        print(x.symbol)

        

    def genapple(self):
        applefields = [xy for xy in self.allfields if xy not in self.snake]
        if len(applefields)==0:
            self.apple = None
        else:
            self.apple = random.choice(applefields)
            diff = abs(self.apple[0]-self.snake[0][0]) + abs(self.apple[1]-self.snake[0][1])
            print("appleMD", len(self.snake), diff, self.apple, self.snake[0])
            self.md += diff
            self.mdList.append(diff)
    

    def mymodel(self,model):
        s = str(model.cost)
        self.cost = int(s[1:-1])-1
        self.tmpoptimize.append(self.cost)
        #print(model.number, model.symbols(shown=True))
        if model.contains(Function("dummy",[])):
            print("dummy", len(self.snake))
        print("solution", len(self.snake), self.cost, (timeit.default_timer() - self.tic)*1000, flush=True)
        self.path = [[0,0]]*(self.n*self.m)
        #heurcount = 0
        for a in model.symbols(shown=True):
            if a.name == "path" and len(a.arguments) == 2:
                i = a.arguments[1].number-1
                xy = a.arguments[0]
                x = xy.arguments[0].number
                y = xy.arguments[1].number
                self.path[i] = [x,y]
        # for a in model.symbols(atoms=True):
        #     if a.name == "heur" and len(a.arguments) == 2 and a.positive :
        #         heurcount += 1
        self.path = rotatepath(self.path)
        self.solution = len(self.snake)
        #print("model#", model.number)

    def eatapple(self):
        #todo:fallback
        #print(apple,path)
        if self.path[0] == self.snake[0]:
            #self.snakevis.printsnake(self.snake, self.apple, self.path)
            #self.snake = [self.path[0]]+self.snake[:-1]
            self.path = rotatepath(self.path)
            print("No_new_path_generated", len(self.snake))
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
                #print("starvingSnake")
                break
            if self.path[0] in self.snake[:-1]:
                # if self.mixup:
                #     self.mixup = False
                #     if len(self.snake)==self.n*self.m:
                #         break

                # if len(self.snake)<self.n*self.m:s
                    self.boobytrap = len(self.snake)
                    print("SnakeBitItself", self.path[0], self.snake[:-1], flush=True)
                    break
            self.snake = [self.path[0]]+self.snake[:-1]
            self.path = rotatepath(self.path)
            self.appleSteps += 1
            #print("rotate", path)

        self.snakevis.printsnake(self.snake, self.apple, self.path, self.rotate)
        self.snakevis.nextiter()
        if len(self.path)>0 and not self.skip:
            self.snake = [self.path[0]]+self.snake
        self.skip = False

        #--------------mirror------------
        if self.rotate!=0:
            snake =  copy.deepcopy(self.snake)
            path =  copy.deepcopy(self.path)
            rotate = copy.deepcopy(self.rotate)
            #print("-----",self.snake,self.rotate )
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
            self.rotate =  copy.deepcopy(rotate)
            #print("-----",self.snake,self.rotate)

            


    def notend(self):
        #print(self.snake,self.apple)
        return len(self.snake) <= self.n*self.m and \
                self.apple is not None and \
                self.boobytrap == 0 and \
                self.notgen == False and \
                self.unsat == [] 
    

    def assignListDoubleBrackets(self, predicate, xy1, xy2, val=True):
        #print("assign", Function(predicate, list2Number(xy)), val)
        #x = list2Number(xy)
        
        #tmp1 = Function("",[Number(xy1[0]),Number(xy1[1])])
        #tmp2 = Function("",[Number(xy2[0]),Number(xy2[1])])
        #self.ctl.assign_external(Function(predicate, [tmp1,tmp2]), val) 
        self.ctl.assign_external(Function(predicate,  [Function("",list2Number(xy1)), Function("",list2Number(xy2))]), val) #assign once


        #print(Function(predicate,  [Function("",list2Number(xy1)), Function("",list2Number(xy2))]), val)

    def assignSnakeHeuristic(self, val):
        if val == True:
            if len(self.snake)>1:
                path = self.path
            else:
                path = easyhc(self.n, self.m)
            for (x,y) in pairwise(path+[path[0]]):
                    self.assignListDoubleBrackets("heur", x, y, val)
                    self.heurlist.append(( x, y))
        if val == False:
            for ( x, y) in self.heurlist:
                self.assignListDoubleBrackets("heur", x, y, val)
            self.heurlist = []

                #maxDist = self.path.index(self.apple)+2
                #self.ctl.assign_external(Function("maxdist",  [Number(maxDist)]), val) 



    def assignListBrackets(self, predicate, xy, val=True):
        #print("assign", Function(predicate, list2Number(xy)), val)
        #x = list2Number(xy)
        self.ctl.assign_external(Function(predicate,  [Function("",list2Number(xy))]), val) #assign once

    def assignList(self, predicate, xy, val=True):
        #print("assign", Function(predicate, list2Number(xy)), val)
        self.ctl.assign_external(Function(predicate, list2Number(xy)), val) #assign once

    def assignLists(self, predicate, xy, val=True):
        for x in xy:
            self.assignList(predicate, x, val)

    def presolve(self):
        self.assignListBrackets("apple", self.apple, True)
        #self.assignList("head", self.snake[0], True)
        #if strat[self.strategy] == strat["hybrid"] and not self.name.startswith("o"):
        #    self.assignLists("mark", self.snake, True)
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
                    print("timeout", len(self.snake), self.cost, flush=True)
                    break  
            if handle.get().exhausted and not handle.get().unsatisfiable:
                print("optimum", len(self.snake), self.cost, (timeit.default_timer() - self.tic)*1000, flush=True)
                if self.optimal == len(self.snake)-1:
                    self.optimal += 1
            if handle.get().unsatisfiable == True and len(self.snake)+1 < self.n*self.m:
                print("UNSAT",len(self.snake),self.solution)
                self.unsat.append((len(self.snake),self.apple,self.snake,self.path))
                self.skip = True


    def postsolve(self):
        print("apple", self.apple, self.n, self.m, self.cost)
        print("snake", self.snake)
        print("path", self.path, flush=True)
        self.assignListBrackets("apple", self.apple, False)
        self.assignListBrackets("head", self.snake[0], False)
        self.assignSnakeHeuristic(False)
        #if strat[self.strategy] == strat["hybrid"] and not self.name.startswith("o"):
        #    self.assignLists("mark", self.snake, False)
        self.ctl.cleanup

        self.toc = timeit.default_timer()
        self.timesolve += (self.toc - self.tic)
        self.steps += self.cost
        self.stepList.append(self.cost)
        self.timegroundList.append(self.timeground)
        self.timesolveList.append(self.timesolve)
        self.optimizeList.append(self.tmpoptimize)
        if self.solution < len(self.snake): # valid because the snake did not change yet, only the path
            print( "notfound",len(self.snake),self.solution)

        print( "finish_search",len(self.snake),self.cost,(self.toc - self.tic)*1000, flush=True)


    def searchpath(self):
        self.presolve()
        self.handlesolve()
        self.postsolve()
        return len(self.path)>0
    

    
class ms_nogood(snakeC):
    def __init__(self,n,m,to,grafik,strategy, rotate):
        self.name = "ms_nogood"
        self.initground = [("nogoodBase",[]), ("extApples",[]) , ("extHeadsMirror",[]),  ("base",[])] #("redoBase",[])
        if rotate==0:
            self.initground.append(("extHeads",[])) 
        #if strat[strategy] in [strat["shortcut"],strat["hybrid"]]:
        #    self.initground.append(("assume",[]))
        super().__init__( n,m,to,grafik,strategy, rotate)



    def handlesolve(self):
        i = 0
        while not self.timeout:
            super().handlesolve()
            if self.mixup == False or self.modelnr>1 or i>10 or self.timeout:
                if self.timeout:
                    self.mixupResolved += "t"
                if self.mixup == True and self.modelnr<2:
                    print("CouldNotResolveMixup", len(self.snake))
                    #self.important.append(len(self.snake))
                    self.mixupResolved += "!"
                else:
                    self.mixupResolved += "."
                break
            self.mixupResolved += "?"
            print("MixupTryToFix", len(self.snake), i, self.mixupResolved)
            self.mixup = False
            self.ctl.ground([("ignore",[Number(random.randint(0,999999))])]) 
            i = i+1



    def setNoPath(self, model, snake, predicate = "path", val=True): # shortcut, hybrid
        l = len(snake) 
        for i,xy in enumerate(snake):
            if i == 0:
                continue
            if i == 1 and l==2:  # snake can not switch position with neck for length 2
                for j in range(2,l-i+2):
                    model.context.add_nogood( [(Function(predicate, [ Function("", [ Number(xy[0]),Number(xy[1])])   ,Number(j)]), val)])
            else:
                if l-i-1>0:     # 1 .. len-1
                    for j in range(2,l-i+2-1): # head is at position 1, therefore paths starts at 2
                        model.context.add_nogood( [(Function(predicate, [ Function("", [ Number(xy[0]),Number(xy[1])])   ,Number(j)]), val)])

    def setHybrid(self, model, snake): # hybrid
        rsnake = snake[::-1]
        cl=[]
        a=-6
        for i in range(len(snake)-1):
            s = i+1

            #print("nogoog", "xys", rsnake[i], s, " to ", rsnake[i+1], "A ",self.mdList[-1]+1,".." ,s+2-1, end=": ")
            for a in range(self.mdList[-1]+1,s+2):
                # :- not next(X,Y), appleN(A), snakeI(X,S), S+2>A, snakeI(Y,S+1).
                tmp =  list2Number(rsnake[i] + rsnake[i+1])   # reverse order 
                xy = Function("",[tmp[0],tmp[1]])
                xynext = Function("",[tmp[2],tmp[3]])
                cl = []
                cl.append((Function("appleN", [ Number(a)]), True))
                cl.append((Function("next", [xy,xynext]), False))
                #print(StringTransformer().transform(str(cl)))
                model.context.add_nogood( cl)
                #print(a, end=",")
            #print()
        #print("snake", self.snake)
        
        #print(cl)
           
    def setnexts(self, model, snake, predicate = "next", val=True): # conservative
        for i in range(len(snake) - 1):
            tmp =  list2Number(snake[i+1] + snake[i])   # reverse order 
            tmp1 = Function("",[tmp[0],tmp[1]])
            tmp2 = Function("",[tmp[2],tmp[3]])
            model.context.add_clause( [(Function(predicate, [tmp1,tmp2]), val)])

    def mymodel(self,model):
        self.modelnr = model.number
        if model.number == 1: #model.contains(Function("dummy",[])):
            if not model.contains(Function("dummy",[])):
                print("MIXUP dummy > 1",model.number, model.cost, len(self.snake))
                self.mixup = True
                self.mixupResolved += "x"
            else:
                print("dummy", len(self.snake))
            #print(model.number, model.symbols(shown=True))
            if strat[self.strategy] in [strat["shortcut"],strat["hybrid"]]:
                self.setNoPath( model, self.snake, "path", True)
            if strat[self.strategy] == strat["hybrid"]:
                self.setHybrid( model, self.snake)
            if strat[self.strategy] == strat["conservative"]:
                self.setnexts( model, self.snake, "next", True)
        else:
            if model.contains(Function("dummy",[])):
                #self.mixupResolved += ":"
                print("MIXUP 1, not dummy",model.number,model.cost,  len(self.snake))
            super().mymodel( model)

class ms_assume(snakeC):
    def __init__(self,n,m,to,grafik,strategy, rotate):
        self.name = "ms_assume"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]), ("base",[]), ("assume",[]) ]
        if rotate==0:
            self.initground.append(("extHeads",[])) 
        if strat[strategy] == strat["hybrid"]:
            self.initground.append(("assumeMark",[]))

        super().__init__( n,m,to,grafik,strategy, rotate)

    def assumeSmaller(self, snake, predicate = "smaller", val=True):
        assumption = []
        l = len(snake) 
        for i,xy in enumerate(snake):
            if i == 0:
                continue
            if i == 1 and l==2:
                #print("assume", predicate,[Number(xy[0]),Number(xy[1]),Number(l-i)], val)
                assumption.append((
                    Function(predicate, [
                             Function("",[Number(xy[0]),Number(xy[1])]),
                             Number(l-i+2)
                             ]), val))
            else:
                if l-i-1>0:
                    #print("assign", predicate,[Number(xy[0]),Number(xy[1]),Number(l-i-1)], val)
                    assumption.append((
                        Function(predicate, [
                                 Function("",[Number(xy[0]),Number(xy[1])]),
                                 Number(l-i-1+2)
                                 ]), val))
        return assumption
    
    def assumeSnake(self, snake, predicate = "snake", val=True):
        assumption = []
        l = len(snake) 
        for i,xy in enumerate(snake):
            assumption.append((Function(predicate, [
                Function("",[Number(xy[0]),Number(xy[1])]),
                Number(l-i)
                ]), val))
        return assumption
    
    def assumeNext(self, snake, predicate = "next", val=True):
        assumption = []
        for i in range(len(snake) - 1):
            tmp =  list2Number(snake[i+1] + snake[i])   # reverse order 
            tmp1 = Function("",[tmp[0],tmp[1]])
            tmp2 = Function("",[tmp[2],tmp[3]])
            assumption.append((Function(predicate, [tmp1,tmp2]), val))
        return assumption

    def presolve(self):
        if strat[self.strategy] == strat["shortcut"]:
            self.assume = self.assumeSmaller(self.snake, "smaller", False)
        if strat[self.strategy] == strat["hybrid"]:
            self.assume = self.assumeSmaller(self.snake, "smaller", False)
            self.assume.extend(self.assumeSnake(self.snake, "snake", True))
        if strat[self.strategy] == strat["conservative"]:
            self.assume = self.assumeNext(self.snake, "next", True)
        #print("assume",self.assume)
        super().presolve()

class ms_preground(snakeC):
    def __init__(self,n,m,to,grafik,strategy, rotate):
        self.name = "ms_preground"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]), ("base",[]) ] 
        if rotate==0:
            self.initground.append(("extHeads",[])) 
        if strat[strategy] == strat["shortcut"]:
            self.initground.append(("consPre",[]))
        if strat[strategy] == strat["hybrid"]:
            self.initground.append(("consPre",[]))
            self.initground.append(("consHybrid",[]))
        if strat[strategy] == strat["conservative"]:
            self.initground.append(("consNaive",[]))
        super().__init__( n,m,to,grafik,strategy, rotate)

    def assignExts(self, snake, predicate = "smaller", val=True):
        l = len(snake) 
        for i,xy in enumerate(snake):
            if i == 0:
                continue
            if i == 1 and l==2:
                #print("assign", predicate,[Number(xy[0]),Number(xy[1]),Number(l-i)], val)
                self.ctl.assign_external(
                    Function(predicate, [
                        Function("", [Number(xy[0]) , Number(xy[1])]),
                        Number(l-i+2)
                        ]), val)
                #print("   smaller", xy[0], xy[1], l-i+2, val)
            else:
                if l-i-1>0:
                    #print("assign", predicate,[Number(xy[0]),Number(xy[1]),Number(l-i-1)], val)
                    self.ctl.assign_external(
                        Function(predicate, [
                            Function("",[Number(xy[0]),Number(xy[1])]),
                            Number(l-i-1+2)
                            ]), val)
                    #print("   smaller", xy[0], xy[1], l-i-1+2, val)

    def assignExtsSnake(self, snake, predicate = "snakeInv", val=True):
        l = len(snake) 
        for i,xy in enumerate(snake):
            #print("assign",predicate, [Number(xy[0]),Number(xy[1]),Number(l-i)],val)
            self.ctl.assign_external(Function(predicate, [Number(xy[0]),Number(xy[1]),Number(l-i)]), val)
            #print("   snakeInv", xy[0], xy[1], l-i)
            
    def assignNextExts(self, snake, predicate = "nextext", val=True):
        for i in range(len(snake) - 1):
            tmp = snake[i+1] + snake[i] # reverse order
            #print("set", Function(predicate, list2Number(tmp)), val)
            self.ctl.assign_external(Function(predicate, list2Number(tmp)), val)
            #print("   nextext", list2Number(tmp), val)

    def assignHybrid(self, snake, predicate = "nextext", val=True):
        for i in range(len(snake) - 1):
            tmp = snake[i+1] + snake[i] + [len(snake)-(i+1)] # reverse order
            #print("set", Function(predicate, list2Number(tmp)), val)
            self.ctl.assign_external(Function(predicate, list2Number(tmp)), val)
            #print("   nextextLen", list2Number(tmp), val)
        

    def presolve(self):
        if strat[self.strategy] in [strat["shortcut"]]:
            self.assignExts(self.snake, "smaller", True)
        if strat[self.strategy] == strat["hybrid"]:
        #    self.assignExtsSnake(self.snake, "snakeInv", True)
            self.assignHybrid(self.snake, "nextext", True)
            self.assignExts(self.snake, "smaller", True)
        if strat[self.strategy] == strat["conservative"]:
            self.assignNextExts(self.snake, "nextext", True)
        super().presolve()
        
    def postsolve(self):
        if strat[self.strategy] in [strat["shortcut"]]:
            self.assignExts(self.snake, "smaller", False)
        if strat[self.strategy] == strat["hybrid"]:
            #self.assignExtsSnake(self.snake, "snakeInv", False)
            self.assignHybrid(self.snake, "nextext", False)
            self.assignExts(self.snake, "smaller", False)
        if strat[self.strategy] == strat["conservative"]:
            self.assignNextExts(self.snake, "nextext", False)
        super().postsolve()
        
class ms_redo(snakeC):
    def __init__(self,n,m,to,grafik,strategy, rotate):
        self.name = "ms_redo"
        self.initground = [("extApples",[]) , ("extHeadsMirror",[]), ("base",[]) ] 
        if rotate==0:
            self.initground.append(("extHeads",[])) 
        super().__init__( n,m,to,grafik,strategy, rotate)

    def snake2cons(self, snake, predicate = "consRedo"):
        toground =[]
        l = len(snake) 
        for i,xy in enumerate(snake):
            if i == 0:
                continue
            if i == 1 and l==2:
                toground.append((predicate,[Number(xy[0]),Number(xy[1]),Number(l-i+2),Number(l)]))
            else:
                if l-i-1>0:
                    toground.append((predicate,[Number(xy[0]),Number(xy[1]),Number(l-i-1+2),Number(l)]))
        return toground
    
    # def snake2mark(self, snake, predicate = "redoHybrid"):
    #     toground =[]
    #     l = len(snake) 
    #     for i,xy in enumerate(snake):
    #             toground.append((predicate,[Number(xy[0]),Number(xy[1]),Number(l-i),Number(l)]))
    #     return toground
    

    def snake2mark(self, snake, predicate = "redoHybrid"):
        toground =[]
        l = len(snake) 
        isnake = snake[::-1]
        for i,xy in enumerate(isnake[:-1]):
            toground.append((predicate,[Number(xy[0]),Number(xy[1]),Number(isnake[i+1][0]),Number(isnake[i+1][1]),Number(i+1),Number(l)]))
        return toground
    
    def snake2next(self, snake, predicate = "consRedo"):
        toground =[]
        l = len(snake)
        for i in range(len(snake) - 1):
            tmp = snake[i+1] + snake[i] + [l] #change order!
            toground.append((predicate,list2Number(tmp))) 
        return toground

    def presolve(self):
        toground = [("extRedo",[Number(len(self.snake))])]
        if strat[self.strategy]  == strat["shortcut"]:
            toground.extend(self.snake2cons(self.snake, "consRedo"))
        if strat[self.strategy] == strat["hybrid"]:
            toground.extend(self.snake2cons(self.snake, "consRedo"))
            toground.extend(self.snake2mark(self.snake, "redoHybrid"))
        if strat[self.strategy] == strat["conservative"]:
            toground.extend(self.snake2next(self.snake, "redoNaive"))
        self.tic = timeit.default_timer()
        self.ctl.ground(toground)
        self.toc = timeit.default_timer()
        self.timeground += (self.toc - self.tic)

        print("ground", len(self.snake), self.timeground, (self.toc - self.tic))
        #print("ground", toground)
        self.assignList("redo", [len(self.snake)], True)
        super().presolve()
        
    def postsolve(self):
        self.ctl.release_external(Function("redo", [Number(len(self.snake))]))
        super().postsolve()

class oneshot(snakeC):
    def __init__(self,n,m,to,grafik,strategy, rotate):
        self.name = "oneshot"
        self.initground = []  
        super().__init__( n,m,to,grafik,strategy, rotate)


    def presolve(self):
        self.ctl = Control(self.ctllist)
        self.ctl.load(self.encoding) 
        
        toground = [("base",[]) ]  
        toground.append(apple2symbol(self.apple))
        toground.extend(snake2symbol(self.snake))
        if strat[self.strategy] == strat["conservative"]:
            toground.append(("osNaive",[]))
        if strat[self.strategy] == strat["shortcut"]:
            toground.append(("osShortcut",[]))
        if strat[self.strategy] == strat["hybrid"]:
            toground.append(("markSnake",[]))
            toground.append(("osShortcut",[]))


        #print(self.strategy, strat[self.strategy], toground)
        self.tic = timeit.default_timer()
        self.ctl.ground(toground)
        self.assignSnakeHeuristic( True)
        self.toc = timeit.default_timer()
        self.timeground += (self.toc - self.tic)
        print("ground", len(self.snake), self.timeground, (self.toc - self.tic))
        self.tic = timeit.default_timer()
             


def main():
    tstart = timeit.default_timer()

    classmap = {
			'os':oneshot,
			'redo':ms_redo,
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

    to = 10.0 # seconds
    n=6
    m=6
    grafik = False
    approach = "redo"
    strategy = "conservative" # conservative, shortcut, hybrid
    rotate = 0
    printhelp = f"""
Script to let the computer play snakes! in clingo! using python! and multi-shot!
How to use this script:
python snakes.py M N A S G T
M, N: dimensions of grid, default {n} and {m}
A: approach, default {approach};    {", ".join(classmap.keys()) }
S: strategy, default {strategy};    {", ".join(strat.keys())}
G: draw grafics, 0 for no (default), 1 for yes    
T: timeout in seconds, default {to} 
example:   python snakes.py 8 8 redo hybrid 1
"""
        
    if len(sys.argv) == 1 or  len(sys.argv) > 1 and sys.argv[1] in ["h", "-h", "--h", "--help"] :
        print(printhelp)
        #return
        n   =  6 # n
        m   =  6 # m
        approach  =  "1" # which method
        strategy  =  "1" # which logic program
        grafik  =  0 # draw pictures + animation
        to  =  5 # timeout
        rotate  =  1 # draw pictures + animation


    if len(sys.argv) > 1 and int(sys.argv[1])>1: n   =  int(sys.argv[1]) # n
    if len(sys.argv) > 2 and int(sys.argv[2])>1: m   =  int(sys.argv[2]) # m
    if len(sys.argv) > 3 and len(sys.argv[3])>0: approach  =  sys.argv[3][0].lower() # which method
    if len(sys.argv) > 4 and len(sys.argv[4])>0: strategy  =  sys.argv[4][0].lower() # which logic program
    if len(sys.argv) > 5: grafik  =  int(sys.argv[5]) # draw pictures + animation
    if len(sys.argv) > 6: to  =  float(sys.argv[6]) # timeout
    if len(sys.argv) > 7: rotate  =  int(sys.argv[7]) # draw pictures + animation


    if approach not in classmap:
        print("choose a valid mode (3rd argument) - was ", approach, ", should be one of", ", ".join([key for key in classmap.keys()]))
        print(printhelp)
        return
    if strategy not in strat:
        print("choose a valid stragegy (4th argument) - was", strategy, ", should be one of", ", ".join([key for key in strat.keys()]))
        print(printhelp)
        return
    for s in ["conservative","shortcut","hybrid"]:
        if strat[strategy] == strat[s]: 
            strategy=s
    
    mysnake = classmap[approach](n,m,to,grafik,strategy, rotate)
    #n = ctl.get_const("n").number

    #mysnake.snakevis.printsnake(mysnake.snake,None,[])
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

    #mysnake.eatapple() 

    print("premature",  len(mysnake.unsat), mysnake.unsat)
    print("nmto", mysnake.n, mysnake.m, mysnake.to)
    print("steplist",mysnake.stepList)
    print("mdList",mysnake.mdList)
    print("optimizeList",mysnake.optimizeList)
    print("timegroundList",mysnake.timegroundList)
    print("timesolveList",mysnake.timesolveList)
    print("steps", mysnake.steps, mysnake.md, float( mysnake.steps)/len(mysnake.snake), float( mysnake.md)/len(mysnake.snake), mysnake.m*mysnake.n/2)
    print("applesteps", mysnake.appleSteps)
    print("ttotal", (timeit.default_timer() - tstart))
    print("tground", mysnake.timeground, mysnake.timegroundinit)
    print("tsolve", mysnake.timesolve)
    print("modus", mysnake.name, mysnake.strategy)
    print("finished", n*m, len(mysnake.snake), mysnake.optimal+1)
    if len(mysnake.mixupResolved) > 0:
        print("mixupResolved", mysnake.mixupResolved)
    if len(mysnake.important)>0:
        print("important",mysnake.important )
    mysnake.snakevis.gengif()

if __name__ == "__main__":
    main()
