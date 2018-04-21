import pickle

class chr(object):
    def __init__(self,sup=None,sub=None,next=None,parent=None,index=None,label=None):
        self.sup=sup
        self.sub=sub
        self.next=next
        self.parent=parent
        self.index=index
        self.label=label

# f=open('boxes.pkl','rb')
# x,y,x1,y1=pickle.load(f)
# labels=['(','2','3','+','\\int','5','x','2','-','9','\\theta',')']
def printTree(start):
	if start is None:
		return
	print (start.label,end=' ')
	if start.sub is not None:
		print ('_{',end='')
		printTree(start.sub)
		print ('}',end='')
	if start.sup is not None:
		print ('^{',end='')
		printTree(start.sup)
		print ('}',end='')
	printTree(start.next)

def insert(prev,curr,x,x1,y,y1):
    curr_top = y[curr.index]
    curr_bot = y1[curr.index]
    prev_top = y[prev.index]
    prev_bot = y1[prev.index]
    prev_avg = 0.5 * (y[prev.index] + y1[prev.index])
    prnt=prev.parent
    if prnt is None:
        if curr_bot < prev_avg:
            prev.sup=curr
            curr.parent=prev
            prev=curr
        elif curr_top>prev_avg:
            prev.sub=curr
            curr.parent=prev
            prev=curr
        else:
            prev.next=curr
            curr.parent=prev.parent
            prev=curr
        return prev
    else:
        prnt_avg = 0.5 * (y[prnt.index] + y1[prnt.index])
        if prev_bot<prnt_avg:
            if curr_bot<prnt_avg:
                if curr_bot<prev_avg and 1.05*curr_bot>prev_top:
                    prev.sup = curr
                    curr.parent = prev
                    prev = curr
                elif curr_top>prev_avg:
                    prev.sub = curr
                    curr.parent = prev
                    prev = curr
                elif curr_top<prev_avg and curr_bot>prev_avg:
                    prev.next = curr
                    curr.parent = prev.parent
                    prev = curr
                else:
                    return insert(prnt, curr,x,x1,y,y1)
            else: return insert(prnt,curr,x,x1,y,y1)
        elif prev_top>prnt_avg:
            if curr_top>prnt_avg:
                if curr_bot<prev_avg:
                    prev.sup = curr
                    curr.parent = prev
                    prev = curr
                elif curr_top>prev_avg and curr_top<prev_bot*1.05:
                    prev.sub = curr
                    curr.parent = prev
                    prev = curr
                elif curr_top<prev_avg and curr_bot>prev_avg:
                    prev.next = curr
                    curr.parent = prev.parent
                    prev = curr
                else:
                    return insert(prnt, curr,x,x1,y,y1)
            else : return insert(prnt,curr,x,x1,y,y1)
        else:
            print('case 3 error')
        return prev




# start=chr(index=0,label=labels[0])
# prev=start
# for i in range(1,len(x)):
#     curr=chr(index=i,label=labels[i])
#     prev=insert(prev,curr)
#
# printTree(start)
# for i in range(1,len(x)):
#     prnt=prev.parent
#     prev_avg=0.5*(y[prev.index]+y1[prev.index])
#     curr=chr(index=i)
#     curr_top=y[curr.index]
#     curr_bot=y1[curr.index]
#     if curr_bot<prev_avg:
#         if prnt is not None:
#             prnt_avg=0.5*(y[prnt.index]+y1[prnt.index])
#         else:
#             prnt_avg=curr_top-1
#         if curr_top<prnt_avg:
#             if curr_bot<prnt_avg:
#                 prnt.sup=curr
#                 curr.parent=prnt
#                 prev=curr
#             else:
#                 prnt.next=curr
#                 curr.parent=prnt.parent
#                 prev=curr
#         else:
#             prev.sup=curr
#             curr.parent=prev
#             prev=curr
#     elif curr_top>prev_avg:
#         if prnt is not None:
#             prnt_avg=0.5*(y[prnt.index]+y1[prnt.index])
#         else:
#             prnt_avg=curr_bot+1
#         if curr_bot>prnt_avg:
#             if curr_top>prnt_avg:
#                 prnt.sub=curr
#                 curr.parent=prnt
#                 prev=curr
#             else:
#                 prnt.next=curr
#                 curr.parent=prnt.parent
#                 prev=curr
#         else:
#             prev.sub=curr
#             curr.parent=prev
#             prev=curr
#     else:
#         prev.next=curr
#         curr.parent=prev.parent
#         prev=curr

