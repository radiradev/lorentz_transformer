import torch

def lorentz_factor(beta):
    return 1 / torch.sqrt(1-beta**2)


def _sample_four_velocity():
    proper_norm = False
    while not proper_norm:
        # Sample a random 3-velocity vector with components in the range [-1, 1]
        three_velocity = 2 * torch.randn(3) - 1
        
        # Check if the spatial components of the 4-velocity vector satisfy the relativistic normalization condition
        proper_norm = torch.norm(three_velocity) < 1

    # Compute the magnitude of the 3-velocity vector
    beta = torch.norm(three_velocity)
    
    # Compute the Lorentz factor
    gamma = 1 / torch.sqrt(1 - beta**2)
    # Combine the time and spatial components to obtain a proper 4-vector
    four_velocity = torch.hstack([gamma, gamma * three_velocity])
    return four_velocity

def sample_four_velocity(n_samples):
    coors = torch.vstack(
        [_sample_four_velocity() for _ in range(n_samples)]
    ).unsqueeze(0)
    return coors


def boost_x(beta):
    beta = torch.tensor([beta])
    gamma = lorentz_factor(beta)
    return torch.tensor([
        [gamma, -beta*gamma, 0, 0],
        [-beta*gamma, gamma, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=beta.dtype)


def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    return torch.sign(p) * torch.log(torch.abs(p) + 1)
