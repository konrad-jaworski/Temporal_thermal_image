import numpy as np
import random
import torch
import torchvision.transforms.functional as TF


class NoiseAddition:
    """
    Add Gaussian noise to B-scan only, keep depth unchanged.
    """
    def __init__(self, sigma_min=0.15, sigma_max=1.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, bscan, depth):
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        noise = torch.randn_like(bscan) * sigma
        return bscan + noise, depth
    
class RandomHorizontalFlipBscan:
    """
    Randomly flips B-scan data along spatial width (W).
    Applies consistently to:
      - X: [T, W]
      - mask: [W]
      - depth: [W]
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, X, mask):
        """
        Parameters:
        -----------
        X : torch.Tensor
            Shape [T, W]
        mask : torch.Tensor
            Shape [W]

        Returns:
        --------
        X, mask, depth : torch.Tensor
            Possibly flipped tensors
        """
        if random.random() < self.p:
            X = torch.flip(X, dims=[-1])      # flip W
            mask = torch.flip(mask, dims=[-1])
        return X, mask
    
class TwoDefect:
    """
    Class which can produce two defects by flipping the simulation and concatenating
    """
    def __init__(self,p=0.5):
        self.p=p
    
    def __call__(self,X,mask):
        if random.random() < self.p:
            H,W=X.size()
            W_center=W//2
            X_flip=torch.flip(X,dims=[-1])
            mask_flip=torch.flip(mask,dims=[-1])
            if X[:,0].sum()==0:
                X_new=torch.cat((X[:,W_center:],X_flip[:,:W_center]),dim=1)
                mask_new=torch.cat((mask[W_center:],mask_flip[:W_center]),dim=0)
            else:
                X_new=torch.cat((X[:,:W_center],X_flip[:,W_center:]),dim=1)
                mask_new=torch.cat((mask[:W_center],mask_flip[W_center:]),dim=0)
            return X_new, mask_new
        else:
            return X,mask

class HorizontalShift:
    """
    Shift sample horizontally (zero-padded).
    X: (H, W)
    mask: (W,)   (as in your code)
    """
    def __init__(self, p=0.5, min_shift=50, max_shift=256):
        self.p = p
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, X, mask):
        # Always return 3 things
        idx = 0

        if random.random() < self.p:
            H, W = X.size()

            back_ground = torch.zeros_like(X)
            mask_ground = torch.zeros_like(mask)

            # make idx a Python int
            idx = int(torch.randint(self.min_shift, self.max_shift, (1,), device=X.device).item())

            if random.random() < 0.5:
                # shift LEFT: content moves left, zeros on right
                X_shifted = torch.cat((X[:, idx:], back_ground[:, :idx]), dim=1)
                mask_shifted = torch.cat((mask[idx:], mask_ground[:idx]), dim=0)
            else:
                # shift RIGHT: content moves right, zeros on left
                X_shifted = torch.cat((back_ground[:, :idx], X[:, :-idx]), dim=1)
                mask_shifted = torch.cat((mask_ground[:idx], mask[:-idx]), dim=0)

            return X_shifted, mask_shifted

        # no augmentation
        return X, mask

class RandomGaussianBlur:
    def __init__(self, p=0.5, kernel_sizes=(3,5,7,9), sigma_range=(0.5, 3.0)):
        self.p = p
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range

    def __call__(self, X, mask):
        if random.random() < self.p:
            # X must be (C,H,W)
            if X.dim() == 2:
                X = X.unsqueeze(0)

            k = random.choice(self.kernel_sizes)
            sigma = random.uniform(*self.sigma_range)

            X = TF.gaussian_blur(X, kernel_size=k, sigma=sigma)

        return X.squeeze(0), mask
    

class DataNormalization:
    """
    Normalize and denormalize data based on provided min and max values.
    """
    def __init__(self,deltaT_min,deltaT_max):
        self.deltaT_min=deltaT_min
        self.deltaT_max=deltaT_max

    def normalize(self, data):
        return (data - self.deltaT_min) / (self.deltaT_max - self.deltaT_min)
    
    def denormalize(self, data_normalized):
        return data_normalized * (self.deltaT_max - self.deltaT_min) + self.deltaT_min
    
class Interpolate:
    """
    Interpolate data to a new size using PyTorch's interpolate function.
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        
        data_interpolated = torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated.squeeze(0)
    
    
class TSR_extrapolation:
    """
    TSR with extrapolation of data up to 2*t_c2 (t_c2 is the characteristic time of cooling of the sample).
    The function returns reconstructed data (Y_hat) and coefficients of the polynomial (a) and extrapolated data (Y_extra).
    """
    def __init__(self):
        pass

    def converttofloat(self,value,double=True):
        if double:
            y=np.array(value,dtype=np.float64)
        else:
            y=np.array(value,dtype=np.float32)
        return y

    def __call__(self, data,ord=2,extrapolate=True):
        """
        Parameters:
        -----------
        data: dictionary with keys 'data' and 'meta' containing the thermal imaging data
        ord: order of the polynomial used for reconstruction and extrapolation (default is 2)
        extrapolate: whether to perform extrapolation of data (default is True)

        Returns:
        Y_hat: reconstructed data
        a: coefficients of the polynomial
        Y_extra: extrapolated data (only if extrapolate=True)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Y=data['data']
        fps=self.converttofloat(data['meta'][9][1]) # FPS 
        thickness=self.converttofloat(data['meta'][19][1]) # Thickness of a sample
        k=self.converttofloat(data['meta'][25][1]) # Thermal conductivity
        density=self.converttofloat(data['meta'][27][1]) # Density
        sheat=self.converttofloat(data['meta'][29][1]) # Specific heat of a sample

        # We find the point of cooling and max temperature
        idx=np.argmax(Y,axis=0)
        idx=idx[0][0] # just assumption that for all the instances higiest temperature is at the same frame
        # We extract length of the 
        L,H,W=Y[idx:,:,:].shape

        # Removal of the baseline (assuming that the data has 5 frames corresponding to room temperature)
        mean_to_sub=Y[:5].mean()
        Y=Y-mean_to_sub
        Y_centered=Y
        
        # Moving to log scale
        Y=torch.from_numpy(Y).to(torch.double)
        Y=Y.to(device)
        Y_log=torch.log(Y)

        # Shortening and movig it to the device
        Y2=torch.reshape(Y_log[idx:,:,:],(L,H*W)) # L x ( H x W )
        
        # range of polynomials investigated (We only investigate up to 8 order of the polynomial)
        p=torch.arange(3,9)
        
        # Create log time characteristic

        t = torch.arange(1, L+1, device=device, dtype=torch.double)  # 1..L
        x_log = torch.log(t)
        
        # Generate vandermonde matrix 
        V=torch.vander(x_log,ord,increasing=True)
        
        # Generate coeficient vector 
        a=torch.linalg.pinv(V) @ Y2

        # Reconstruction of the Y matrix
        Y_hat=V @ a

        Y_hat=torch.exp(Y_hat)
        Y_hat=torch.reshape(Y_hat,(L,H,W))

        # Conditiononing for extrapolation

        if extrapolate==False:
            return Y_hat,a

        # Formulate new time vector 
        t_c2=2*(thickness**2/(k/(density*sheat)))
        ft_c2=np.floor(fps*t_c2)
        x_p=torch.arange(1,ft_c2+1,device=device,dtype=torch.double) # 1..L_prime
        x_log_p=torch.log(x_p)

        # New Vandermond matrix
        V_p=torch.vander(x_log_p,ord,increasing=True)

        # Extrapolated data
        Y_extra=V_p @ a

        Y_extra=torch.exp(Y_extra)
        Y_extra=torch.reshape(Y_extra,(-1,H,W))

        return Y_hat,a,Y_extra