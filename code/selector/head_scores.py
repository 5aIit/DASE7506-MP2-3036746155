"""Head scores for the frozen selector."""

def head_scores(head,x):
    return head.decision_function(x.reshape(-1,x.shape[-1])).reshape(-1,4)
