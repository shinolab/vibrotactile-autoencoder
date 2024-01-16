import numpy as np
import pickle
import torch

device = torch.device("cuda")
print(f'Selected device: {device}')

def img_denormalize(img):
    # Min of original data: -80
    # Max of original data: 0
    origin_max = 0.
    origin_min = -80.
    img = (img + 1) / 2 # from [-1, 1] back to [0, 1]
    denormalized_img = img * (origin_max - origin_min) + origin_min
    return denormalized_img

def z_denormalize(z):
    # range of real latent space: [-7.24, 6.42]
    origin_max = 6.42
    origin_min = -7.24
    z = (z + 1) / 2
    denormalized_z = z * (origin_max - origin_min) + origin_min
    return denormalized_z

def myFunc(decoder, zs):
    zs = z_denormalize(zs)
    zs = torch.tensor(zs).to(torch.float32).to(device)
    output = img_denormalize(decoder(zs)).reshape(zs.shape[0], -1)
    # output = decoder(zs).reshape(zs.shape[0], -1)
    return output

# def myGoodness(target, xs):
#     xs = torch.tensor(xs).to(device)
#     return np.sum((xs.reshape(xs.shape[0], -1) - target.reshape(1, -1)).cpu().detach().numpy() ** 2, axis=1) ** 0.5

def myGoodness(xs):
    pass

def myJacobian(model, z):
    z = z_denormalize(z)
    z = torch.tensor(z).to(torch.float32).to(device)
    return model.calc_model_gradient(z, device)

def getSliderLength(n, boundary_range, ratio, sample_num=1000):
    samples = np.random.uniform(low=-boundary_range, high=boundary_range, size=(sample_num, 2, n))
    distances = np.linalg.norm(samples[:, 0, :] - samples[:, 1, :], axis=1)
    average_distance = np.average(distances)
    return ratio * average_distance

def getRandomAMatrix(high_dim, dim, optimals, range):
    A = np.random.normal(size=(high_dim, dim))
    try:
        invA = np.linalg.pinv(A)
    except:
        print("Inverse failed!")
        return None

    low_optimals = np.matmul(invA, optimals.T).T
    conditions = (low_optimals < range) & (low_optimals > -range)
    conditions = np.all(conditions, axis=1)
    if np.any(conditions):
        return A
    else:
        print("A matrix is not qualified. Resampling......")
        return None

def SelectVec(z, rank):
    # with open('CAAE_14class/latent_dict.pickle', 'rb') as file:
    #     latent_dict = pickle.load(file)
    
    # avg_dis = 0
    # dis_num = 0
    # # Calculate the average distance of feature
    # for i in range(len(latent_dict['z'])):
    #      for j in range(i+1, len(latent_dict['z'])):
    #           dis = np.linalg.norm(np.array(latent_dict['z'][i]) - np.array(latent_dict['z'][j]))
    #           avg_dis += dis
    #           dis_num += 1
    
    # avg_dis /= dis_num
    # print(avg_dis)
    avg_dis = 14.095

    with open('CAAE_14class/latent_dict.pickle', 'rb') as file:
        latent_dict = pickle.load(file)

    new_z = []
    # No.0 Good
    # Select new_z nearer than avg_dis / 4
    if rank == 0:
        while True:
            index = np.random.randint(len(latent_dict['z']))
            new_z = latent_dict['z'][index]
            dis = np.linalg.norm(np.array(z) - np.array(new_z))
            if dis <= avg_dis / 8:
                break
    # No.1 So-so
    # Select new_z nearer than avg_dis / 2
    elif rank == 1:
        while True:
            index = np.random.randint(len(latent_dict['z']))
            new_z = latent_dict['z'][index]
            dis = np.linalg.norm(np.array(z) - np.array(new_z))
            if dis <= avg_dis / 4:
                break
    # No.2 Bad
    # Select new_z farther than avg_dis
    elif rank == 2:
        while True:
            index = np.random.randint(len(latent_dict['z']))
            new_z = latent_dict['z'][index]
            dis = np.linalg.norm(np.array(z) - np.array(new_z))
            if dis >= avg_dis / 4:
                break

    return new_z


