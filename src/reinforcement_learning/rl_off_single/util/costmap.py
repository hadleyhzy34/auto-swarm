import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

def state2costmap(state):
    """
    description:
    args:
        state: np.array, (scan.ranges, rel_goal_pos), (362,)
    return:
        costmap: torch.tensor, (360,256,3)
    """
    dist_increment = 4. / 256
    angle_increment = 2 * torch.pi / 360

    # pdb.set_trace()
    costmap = torch.zeros((360,256,3))  #image type of obs, (h,w,channel)
    for i in range(360):
        if state[i] > 3.5:
            pass
        else:
            #degree
            deg = (i+180)%360
            dist = int(state[i]/dist_increment)
            costmap[deg,dist] = torch.tensor([1.,0.,0.])  #assign red pixel to obstacle

    # pdb.set_trace()
    # assign waypoint
    deg = torch.atan2(torch.tensor([state[-1]]),torch.tensor([state[-2]]))
    deg = int((deg + torch.pi)/ angle_increment)
    # deg = int(deg / angle_increment)
    cur_dist = np.linalg.norm([state[-2], state[-1]])
    dist = int(cur_dist / dist_increment)
    costmap[deg-5:deg+5,dist-5:dist+5] = torch.tensor([1.,1.,1.])

    # plt.axis('equal')
    # plt.imshow(costmap)
    # plt.show()
    # image = preprocess(state)

    return costmap

def batchState2costmap(state):
    """
    description:
    args:
        state: torch.tensor, (b, scan.ranges + rel_goal_pos), (b, 362,)
    return:
        costmap: torch.tensor, (b,360,256,3)
    """
    dist_increment = 4. / 256
    angle_increment = 2 * torch.pi / 360

    # pdb.set_trace()
    batch_size = state.shape[0]
    costmap = torch.zeros((batch_size,360,256,3))  #image type of obs, (h,w,channel)
    for i in range(360):
        if state[i] > 3.5:
            pass
        else:
            #degree
            deg = (i+180)%360
            dist = int(state[i]/dist_increment)
            costmap[deg,dist] = torch.tensor([1.,0.,0.])  #assign red pixel to obstacle

    # pdb.set_trace()
    # assign waypoint
    deg = torch.atan2(torch.tensor([state[-1]]),torch.tensor([state[-2]]))
    deg = int((deg + torch.pi)/ angle_increment)
    # deg = int(deg / angle_increment)
    cur_dist = np.linalg.norm([state[-2], state[-1]])
    dist = int(cur_dist / dist_increment)
    costmap[deg-5:deg+5,dist-5:dist+5] = torch.tensor([1.,1.,1.])

    # plt.axis('equal')
    # plt.imshow(costmap)
    # plt.show()
    # image = preprocess(state)

    return costmap

def preprocess(data):
    """"
    description: lidar data to grid image
    args:
        data: (state_size,)
    return:
        map: (224,224), torch.floatTensor
    """
    # pdb.set_trace()
    # data = torch.tensor(data,dtype=torch.float,device=self.device)[0]  #(scan_size,)
    
    for i in range(360):
        if data[i] > 3.5:
            data[i]=3.5
    # pdb.set_trace()
    rad_points = torch.zeros((360, 2))  #(360,2)
    rad_points[:,0] = torch.cos((torch.arange(0,360)) * torch.pi / 180) * data[0:360]
    rad_points[:,1] = torch.sin((torch.arange(0,360)) * torch.pi / 180) * data[0:360]
    
    # plt.figure()
    plt.axis('equal')
    plt.scatter(rad_points[:,0].cpu().numpy(),rad_points[:,1].cpu().numpy())
    plt.show()
    # plt.savefig('test1.png')
    print(f'check rad_points here')
    #voxelize 2d lidar points
    rad_points[:,0] -= -3.5
    rad_points[:,1] = 3.5 - rad_points[:,1]
    rad_points = rad_points.div((3.5*2)/224,rounding_mode='trunc').long()
    print(f'check rad after assign: {rad_points.shape}')

    # pdb.set_trace()
    image = torch.empty((224,224))  #(224,224)
    image[:] = 0.
    print(f'check img after assign: {image.shape}')
    image[rad_points[:,0],rad_points[:,1]] = 1.
    
    # remove center point
    image[112,112] = 0.
    print(f'check img here')
    # plt.figure()
    # plt.imshow(image.numpy())
    # plt.show()
    # plt.savefig('test2.png')
    
    return image
 
