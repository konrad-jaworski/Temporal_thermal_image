import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import imageio
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt


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
    
# Validation metrics: -----------------------------------------------------------------------------------

def fit_plateau_1d(pred, threshold=0.03, method='trim'):
    pred = pred.clone()

    mask = pred > threshold

    if mask.sum() == 0:
        plateau = torch.zeros_like(pred)
        return plateau

    coords = mask.nonzero(as_tuple=True)[0]

    best_loss = float("inf")
    best_params = None
    best_plateau = None
    best_c = None  

    for shrink in range(0, 20):

        x_min = coords.min() + shrink
        x_max = coords.max() - shrink

        if x_min >= x_max:
            continue

        segment = pred[x_min:x_max+1]

        # ---- estimator selection ----
        if method == 'mean':
            c = segment.mean()

        elif method == 'median':
            c = segment.median()

        elif method == 'q25':
            c = torch.quantile(segment, 0.25)

        elif method == 'q75':
            c = torch.quantile(segment, 0.75)

        elif method == 'trim':
            sorted_vals, _ = torch.sort(segment)
            n = len(sorted_vals)
            k = int(0.1 * n)

            if n > 2 * k:
                trimmed = sorted_vals[k:-k]
            else:
                trimmed = sorted_vals

            c = trimmed.median()

        else:
            raise ValueError(f"Unknown method: {method}")

        plateau = torch.zeros_like(pred)
        plateau[x_min:x_max+1] = c

        loss = ((pred - plateau)**2).mean()

        if loss < best_loss:
            best_loss = loss
            best_params = (x_min.item(), x_max.item())
            best_plateau = plateau
            best_c = c

    if best_plateau is None:
        x_min = coords.min()
        x_max = coords.max()
        segment = pred[x_min:x_max+1]

        best_c = segment.median()
        best_plateau = torch.zeros_like(pred)
        best_plateau[x_min:x_max+1] = best_c
        best_params = (x_min.item(), x_max.item())

    return best_plateau

def visMet(pred_all,mask_all,step_to,thickness,visualization=True,break_index=None,display_time=1,mode='trim'):
    step=step_to
    N = pred_all.shape[0]
    # Detaching tensors from the graph speed/repro
    pred_all_detach = pred_all.detach()
    gt_all_detach = mask_all.detach()

    # Accumulation for global error
    mae_glob_list=[]
    medae_glob_list=[]
    rmse_glob_list=[]
    
    # Accumulators for roi metric
    mae_roi_list = []
    medae_roi_list = []
    rmse_roi_list = []

    # Accumulation for baground
    mae_bg_list = []            # false positives in sound region

    # single-depth proxy: Method selection
    abs_err_defect_list = []   
    abs_err_defect_list_mean = []
    abs_err_defect_list_25 = []
    abs_err_defect_list_75 = []

    

    # This we are using to plot specific cases over simulations
    index_for_breaking=0

    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + step, N)

            # One simulation at a time
            pred_t = pred_all_detach[start:end, :]  # [B, W]
            gt_t   = gt_all_detach[start:end, :]    # [B, W]\

            for i in range(step):
                pred_t[i,:]=fit_plateau_1d(pred_t[i,:],method=mode)

            # ROI from GT (defect region)
            roi = gt_t > 0
            bg  = gt_t == 0
            
            # Global metric
            err_glob=(pred_t-gt_t).abs() # Absolute error over golabal range

            mae_glob = err_glob.mean().item()
            medae_glob = err_glob.median().item()
            rmse_glob = torch.sqrt(((pred_t-gt_t)**2).mean()).item()

            mae_glob_list.append(mae_glob)
            medae_glob_list.append(medae_glob)
            rmse_glob_list.append(rmse_glob) 

            # ---- Paper metrics (NO mean-filling) ----
            if roi.any():
                # Metric over region of interes
                err_roi = (pred_t[roi] - gt_t[roi]).abs() # Absolute error over ROI

                # General reconstruction of shape and values over roi
                mae_roi  = err_roi.mean().item() # MAE to report
                medae_roi = err_roi.median().item() # MedAE to 
                rmse_roi = torch.sqrt(((pred_t[roi] - gt_t[roi]) ** 2).mean()).item() # Investigation of outliers

                mae_roi_list.append(mae_roi)
                medae_roi_list.append(medae_roi)
                rmse_roi_list.append(rmse_roi)
                
                # --------------------------------------------- Most interesting part------------------
                # Single-number "defect depth" proxy (robust):Median over ROI
                depth_pred_mean=pred_t[roi].mean().item()
                depth_pred=pred_t[roi].median().item()
                depth_pred_25=torch.quantile(pred_t[roi],0.25).item()
                depth_pred_75=torch.quantile(pred_t[roi],0.75).item()


                depth_gt = gt_t[roi].mean().item() # We only have homogeneus values 


                abs_err_defect = abs(depth_pred - depth_gt)
                abs_err_mean = abs(depth_pred_mean - depth_gt)
                abs_err_75 = abs(depth_pred_75 - depth_gt)
                abs_err_25 = abs(depth_pred_25 - depth_gt)
                if gt_t[roi].mean()<0.5:
                    abs_err_defect_list.append(abs_err_defect)
                    abs_err_defect_list_mean.append(abs_err_mean)
                    abs_err_defect_list_25.append(abs_err_25)
                    abs_err_defect_list_75.append(abs_err_75)
            
            else:
                # No defect present in this chunk (if that can happen)
                mae_roi_list.append(np.nan)
                medae_roi_list.append(np.nan)
                rmse_roi_list.append(np.nan)
                abs_err_defect_list.append(np.nan)

            # Metric over background false positive
            if bg.any():
                mae_bg = pred_t[bg].abs().mean().item()
            else:
                mae_bg = np.nan
            mae_bg_list.append(mae_bg)

            # Fixed error scale for visualization only (consistent colors) per chunk
            vis_err = (pred_t - gt_t)
            v = float(vis_err.abs().max())

            if visualization==True:
                pred_np = pred_t.cpu().numpy() # Raw prediction
                gt_np   = gt_t.cpu().numpy() # Gt mask
                err_np  = (pred_t - gt_t).numpy() # We want to show in which areas is overestimated and in which we underestimate
                clear_output(wait=True)
                fig = plt.figure(figsize=(18, 10))

                plt.subplot(2, 2, 1)
                plt.imshow(pred_np, aspect="auto",vmin=0,vmax=1)
                plt.colorbar(shrink=0.7)
                plt.title(f'Prediction rows {start}:{end}')

                plt.subplot(2, 2, 2)
                plt.imshow(gt_np, aspect="auto",vmin=0,vmax=1)
                plt.colorbar(shrink=0.7)
                plt.title(f'Ground truth depth of defect: {gt_np.max():.2f}')

                plt.subplot(2, 2, 3)
                plt.imshow(err_np, cmap='seismic', vmin=-v, vmax=v, aspect="auto")
                plt.colorbar(shrink=0.7)
                plt.title('Error: pred - GT (fixed scale)')

                plt.subplot(2, 2, 4)
                # summary text panel (paper-friendly)
                plt.axis("off")

                # convert normalized errors to mm
                mae_roi_mm = mae_roi_list[-1] * thickness if not np.isnan(mae_roi_list[-1]) else np.nan
                medae_roi_mm = medae_roi_list[-1] * thickness if not np.isnan(medae_roi_list[-1]) else np.nan
                rmse_roi_mm = rmse_roi_list[-1] * thickness if not np.isnan(rmse_roi_list[-1]) else np.nan
                abs_err_defect_mm = abs_err_defect_list[-1] * thickness if not np.isnan(abs_err_defect_list[-1]) else np.nan
                mae_bg_mm = mae_bg_list[-1] * thickness if not np.isnan(mae_bg_list[-1]) else np.nan

                plt.text(
                0.02, 0.98,
                "\n".join([
                    f"ROI MAE (norm):  {mae_roi_list[-1]:.6f}",
                    f"ROI MAE (mm):    {mae_roi_mm:.4f}",
                    f"ROI MedAE (mm):  {medae_roi_mm:.4f}",
                    f"ROI RMSE (mm):   {rmse_roi_mm:.4f}",
                    f"Defect |median(pred)-(gt)| (mm): {abs_err_defect_mm:.4f}",
                    f"BG MAE (mm):     {mae_bg_mm:.4f}",
                ]),
                va="top", ha="left", fontsize=12
                )
                fig.tight_layout()
                plt.show()
                time.sleep(display_time)

            index_for_breaking=index_for_breaking+1
            if index_for_breaking !=None:
                pass
            elif index_for_breaking==break_index:
                break
        
        # -----------------------------
        # Final dataset-level reporting
        # -----------------------------
        def nanmean(x): return float(np.nanmean(np.array(x, dtype=np.float64)))
        def nanmedian(x): return float(np.nanmedian(np.array(x, dtype=np.float64)))

        # Metrics for over all dataset

        mae_glob_norm = nanmean(mae_glob_list)
        medae_glob_norm = nanmedian(medae_glob_list)
        rmse_glob_norm = nanmean(rmse_glob_list)

        mae_roi_norm = nanmean(mae_roi_list)
        medae_roi_norm = nanmedian(medae_roi_list)
        rmse_roi_norm = nanmean(rmse_roi_list)
        mae_bg_norm = nanmean(mae_bg_list)
        abs_err_defect_norm = nanmean(abs_err_defect_list)
        abs_err_defect_mean_norm = nanmean(abs_err_defect_list_mean)
        abs_err_defect_25_norm = nanmean(abs_err_defect_list_25)
        abs_err_defect_75_norm = nanmean(abs_err_defect_list_75)

        print("\n=== PAPER METRICS Global (normalized 0..1) ===")
        print(f"ROI MAE (mean):          {mae_glob_norm:.6f}")
        print(f"ROI MedAE (median):      {medae_glob_norm:.6f}")
        print(f"ROI RMSE (mean):         {rmse_glob_norm:.6f}")

        print("\n=== PAPER METRICS ROI (normalized 0..1) ===")
        print(f"ROI MAE (mean):          {mae_roi_norm:.6f}")
        print(f"ROI MedAE (median):      {medae_roi_norm:.6f}")
        print(f"ROI RMSE (mean):         {rmse_roi_norm:.6f}")
        print(f"BG MAE (mean):           {mae_bg_norm:.6f}")

        print("\n=== Depth prediction over ROI (mm), thickness ===")
        print(f"Defect abs err (median): {abs_err_defect_norm:.6f}")
        print(f"Defect abs err (median): {abs_err_defect_norm * thickness:.4f} mm")
        print(f"Defect abs err (mean): {abs_err_defect_mean_norm * thickness:.4f} mm")
        print(f"Defect abs err (q25): {abs_err_defect_25_norm * thickness:.4f} mm")
        print(f"Defect abs err (q75): {abs_err_defect_75_norm * thickness:.4f} mm")

       