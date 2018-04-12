import pickle

class chr(object):
    def __init__(self,sup=None,sub=None,next=None,parent=None,index=None,label=None):
        self.sup=sup
        self.sub=sub
        self.next=next
        self.parent=parent
        self.index=index
        self.label=label

f=open('boxes.pkl','rb')
x,y,x1,y1=pickle.load(f)

start=chr(index=0)
prev=start
for i in range(1,len(x)):
    prnt=prev.parent
    prev_avg=0.5*(y[prev.index]+y1[prev.index])
    curr=chr(index=i)
    curr_top=y[curr.index]
    curr_bot=y[curr.index]

