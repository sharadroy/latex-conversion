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

def printTree(start):
	if start is None:
		return
	print (start.index)
	if start.sub is not None:
		print ('_{')
		printTree(start.sub)
		print ('}')
	if start.sup is not None:
		print ('^{')
		printTree(start.sup)
		print ('}')
	printTree(start.next)


start=chr(index=0)
prev=start
for i in range(1,len(x)):
    prnt=prev.parent
    prev_avg=0.5*(y[prev.index]+y1[prev.index])
    curr=chr(index=i)
    curr_top=y[curr.index]
    curr_bot=y1[curr.index]
    if curr_bot<prev_avg:
        if prnt is not None:
            prnt_avg=0.5*(y[prnt.index]+y1[prnt.index])
        else:
            prnt_avg=curr_top-1
        if curr_top<prnt_avg:
            if curr_bot<prnt_avg:
                prnt.sup=curr
                curr.parent=prnt
                prev=curr
            else:
                prnt.next=curr
                curr.parent=prnt.parent
                prev=curr
        else:
            prev.sup=curr
            curr.parent=prev
            prev=curr
    elif curr_top>prev_avg:
        if prnt is not None:
            prnt_avg=0.5*(y[prnt.index]+y1[prnt.index])
        else:
            prnt_avg=curr_bot+1
        if curr_bot>prnt_avg:
            if curr_top>prnt_avg:
                prnt.sub=curr
                curr.parent=prnt
                prev=curr
            else:
                prnt.next=curr
                curr.parent=prnt.parent
                prev=curr
        else:
            prev.sub=curr
            curr.parent=prev
            prev=curr
    else:
        prev.next=curr
        curr.parent=prev.parent
        prev=curr

printTree(start)