# from math import dist
import numpy as np
import torch
import pdb

def points2line(points, point_a, point_b):
    """calculate points to line segment distance

    Args:
        points (b,n,2): _description_
        point_a (b,2): _description_
        point_b (b,2): _description_
    return:
        dist: (b,n)
    """
    # pdb.set_trace()
    a2points = points - point_a[:,None,:]  #(b,360,2)
    a2b = point_b - point_a  #(b,2)
    b2points = points - point_b[:,None,:]  #(b,360,2)

    mask_0 = torch.lt(torch.linalg.norm(point_a-point_b,dim=-1)[:,None],0.001)
    distance = mask_0 * torch.linalg.norm(b2points,dim=-1)  #(b,360)

    mask_1 = torch.lt((a2b[:,None,:] * a2points).sum(-1),0) & torch.logical_not(mask_0)
    distance += mask_1 * torch.linalg.norm(a2points,dim=-1)  #(b,360)

    mask_2 = torch.gt((a2b[:,None,:] * b2points).sum(-1),0) & torch.logical_not(mask_0)
    distance += mask_2 * torch.linalg.norm(b2points,dim=-1)  #(b,360)

    mask_3 = torch.logical_not(mask_1) & torch.logical_not(mask_2) & torch.logical_not(mask_0)
    # distance += mask_3 * torch.abs(a2b[:,None,0]*a2points[:,:,1] - a2b[:,None,1]*a2points[:,:,0]) * (1/torch.linalg.norm(a2b[:,None,:],dim=-1))
    # pdb.set_trace()
    distance += mask_3 * torch.abs(a2b[:,None,0]*a2points[:,:,1] - a2b[:,None,1]*a2points[:,:,0]) * (1/torch.linalg.norm(a2b[:,None,:],dim=-1))

    return distance

def p2l(points, point_a, point_b):
    """calculate points to line segment distance

    Args:
        points (b,n,2): non-differentiable
        point_a (b,2): differentiable
        point_b (b,2): non-differentiable
    return:
        dist: (b,n)
    """
    # sampling N points between point a and b
    b = points.shape[0]
    delta_a2b = point_b - point_a  #(b,2)
    distance = None
    weight = torch.arange(0,1,0.05)[None,:].repeat(b,1).cuda()  #(b,20)
    for i in range(20):
        sampling_points = point_a + delta_a2b * weight[:,i][:,None]  #(b,2)
        if distance == None:
            distance = torch.cdist(points, sampling_points[:,None,:])  #(b,360,1)
        else:
            distance = torch.cat([distance, torch.cdist(points, sampling_points[:,None,:])],dim=-1)  #(b,360,i)

    return distance.min(-1).values

    # r = (a2points * a2b[:,None,:]).sum(-1)
    # r = torch.div(r, torch.linalg.norm(a2b,dim=-1)[:,None]**2)

    # mask_1 = (torch.lt(r,1.) & torch.gt(r,0.)).detach()
    # mask_2 = torch.ge(r,1.).detach()
    # mask_3 = torch.le(r,0.).detach()

    # distance = torch.linalg.norm(b2points,dim=-1)*mask_2 + torch.linalg.norm(a2points,dim=-1)*mask_3
    # distance += torch.sqrt(((a2points * a2points).sum(-1) - (r*torch.linalg.norm(a2b,dim=-1)[:,None])**2) * mask_1)

    # return distance

if __name__ == '__main__':
    batch_size = 32
    # points = torch.randn((batch_size,360,2))
    # a = torch.randn((batch_size,2))
    # b = torch.randn((batch_size,2))

    # distance = points2line(points, a, b)

    # pdb.set_trace()
    points = torch.tensor([[-1.,0.],[0.,2.],[2.,0.],[1,3],[2,2],[1,-1],[1,-0.5],[1,0]])
    a = torch.tensor([[0.,0.]])
    b = torch.tensor([[1.,1.]])

    distance = points2line(points,a,b)
    print(distance)