import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class NoiseAddition:
    """
    Add Gaussian noise to B-scan only, keep depth unchanged.
    With modification over the camera response.
    """

    def __init__(self,sigma_min=0.0, sigma_max=0.1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, bscan, depth):
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        noise = torch.randn_like(bscan) * sigma
        return (bscan + noise).clamp_min(0.0), depth
    
class NoiseAdditionExperiment:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, bscan, depth):
        noise = torch.randn_like(bscan) * self.sigma
        bscan_noisy = (bscan + noise).clamp_min(0.0)
        return bscan_noisy, depth
    
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
    Shift sample horizontally using reflect padding.
    X: (H, W)
    mask: (W,)
    """

    def __init__(self, p=0.5, min_shift=1, max_shift=64):
        self.p = p
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, X, mask):
        if random.random() >= self.p:
            return X, mask

        H, W = X.size()
        idx = int(torch.randint(self.min_shift, self.max_shift, (1,)).item())
        direction = random.choice(["left", "right"])

        if direction == "left":
            # pad right, crop left
            X_pad = F.pad(X.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, idx:idx+W]

            mask_pad = F.pad(mask.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[idx:idx+W]

        else:  # right
            # pad left, crop right
            X_pad = F.pad(X.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, :W]

            mask_pad = F.pad(mask.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[:W]

        return X_shifted, mask_shifted

    
class DefectSlopeDropout:
    """
    Drops a horizontal band (partial height) over the defect columns span.
    Biases dropout to early/mid/late parts of the slope.
    """
    def __init__(self, p=0.2, height_frac_range=(0.15, 0.3), width_pad_range=(0, 10), where_probs=(0.33, 0.34, 0.33)):
        self.p = p
        self.height_frac_range = height_frac_range
        self.width_pad_range = width_pad_range
        self.where_probs = where_probs  # (early, mid, late)

    def __call__(self, X, mask):
        if random.random() >= self.p:
            return X, mask

        H, W = X.shape
        idx = torch.where(mask > 0)[0]
        if idx.numel() == 0:
            return X, mask

        left = int(idx[0].item())
        right = int(idx[-1].item())
        pad = random.randint(*self.width_pad_range)
        x0 = max(0, left - pad)
        x1 = min(W, right + pad + 1)

        frac = random.uniform(*self.height_frac_range)
        h = max(1, int(frac * H))

        region_choice = random.choices(["early", "mid", "late"], weights=self.where_probs, k=1)[0]

        def randint_safe(a, b):
            a = int(a); b = int(b)
            if b < a:
                return a
            return random.randint(a, b)

        if region_choice == "early":
            y0 = randint_safe(0, (H // 3) - h)

        elif region_choice == "late":
            y0 = randint_safe(2 * H // 3, H - h)

        else:  # mid
            mid_start = H // 3
            mid_end = (2 * H // 3) - h
            y0 = randint_safe(mid_start, mid_end)

        y0 = max(0, min(y0, H - h))
        y1 = y0 + h

        X_out = X.clone()
        X_out[y0:y1, x0:x1] = 0.0
        return X_out, mask

# Derivatives approximation --------------------------------------------------------------------------------------------------------
def d1_dy(X: torch.Tensor) -> torch.Tensor:
    """
    First derivative along height (H axis / rows).
    X: (H, W) -> (H, W)
    """
    d = torch.zeros_like(X)
    d[1:-1, :] = 0.5 * (X[2:, :] - X[:-2, :])  # central difference
    d[0, :]    = X[1, :] - X[0, :]             # forward difference
    d[-1, :]   = X[-1, :] - X[-2, :]           # backward difference
    return d

def d2_dy2(X: torch.Tensor) -> torch.Tensor:
    """
    Second derivative along height (H axis / rows).
    X: (H, W) -> (H, W)
    """
    d2 = torch.zeros_like(X)
    d2[1:-1, :] = X[2:, :] - 2 * X[1:-1, :] + X[:-2, :]
    d2[0, :]    = d2[1, :]
    d2[-1, :]   = d2[-2, :]
    return d2


def d1_dx(X: torch.Tensor) -> torch.Tensor:
    """
    First derivative along width (spatial axis).
    X: (H_time, W_space)
    """
    d = torch.zeros_like(X)

    # central difference
    d[:, 1:-1] = (X[:, 2:] - X[:, :-2]) * 0.5

    # boundaries
    d[:, 0]  = X[:, 1] - X[:, 0]
    d[:, -1] = X[:, -1] - X[:, -2]

    return d



def d2_dx2(X: torch.Tensor)->torch.Tensor:
    """
    Second derivative along width (W axis/ columns)
    """
    d2 = torch.zeros_like(X)
    d2[:, 1:-1] = X[:,2:] - 2 * X[ :,1:-1] + X[:, :-2]
    d2[:, 0] = d2[:, 1] 
    d2[:, -1]= d2[:, -2]

    return d2
    
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

    def __call__(self, data,ord=3,extrapolate=True):
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
        if Y[:5].mean()<Y[:-5].mean():
            mean_to_sub=Y[:5].mean()
            Y=Y-mean_to_sub
        else:
            print("Baseline is higher than the rest of the data, check the data and remove baseline manually if needed")
        
        # Moving to log scale
        Y=torch.from_numpy(Y).to(torch.double)
        Y=Y.to(device)
        Y_log=torch.log(Y)

        # Shortening and movig it to the device
        Y2=torch.reshape(Y_log[idx:,:,:],(L,H*W)) # L x ( H x W )
        
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
        ft_c2=np.round(fps*t_c2)
        x_p=torch.arange(1,ft_c2+1,device=device,dtype=torch.double) # 1..L_prime
        x_log_p=torch.log(x_p)

        # New Vandermond matrix
        V_p=torch.vander(x_log_p,ord,increasing=True)

        # Extrapolated data
        Y_extra=V_p @ a

        Y_extra=torch.exp(Y_extra)
        Y_extra=torch.reshape(Y_extra,(-1,H,W))

        # Correction of the first frame of the extrapolated data to match the first frame of the original data
        if Y_extra[0,:,:].mean()>Y[idx,:,:].mean():
            Y_extra[0,:,:]=Y[idx,:,:]
        

        return Y_hat,a,Y_extra