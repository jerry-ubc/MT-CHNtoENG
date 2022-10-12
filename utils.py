import torch

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("<ea> saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("<ea> loading checkpoing")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #if adding other parameters, use "checkpoint['best_curr_accuracy']" for parameter "best_curr_accuracy"