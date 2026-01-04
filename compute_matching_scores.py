
import os
import sys
import torch
import numpy as np
import PIL.Image
from typing import Tuple, List, Optional

# Add mast3r root to path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# If mast3r package is in /home/yashturkar/Workspace/mast3r/, then that dir is the root.
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Imports - let them fail if missing to aid debugging
import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import extract_correspondences_nonsym, fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

class MASt3RModel:
    _instance = None
    _device = None

    @classmethod
    def get_instance(cls, device='cuda', model_name="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"):
        if cls._instance is None:
            # print(f"Loading MASt3R model: {model_name} on {device}")
            # Notebook uses: "naver/{model_name}"
            # model_name default in notebook: "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            full_name = f"naver/{model_name}"
            model = AsymmetricMASt3R.from_pretrained(full_name).to(device)
            model.eval()
            cls._instance = model
            cls._device = device
        return cls._instance

def to_numpy_safe(tensor):
    if isinstance(tensor, torch.Tensor): 
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def compute_matching_scores(
    img1_path: str, 
    img2_path: str, 
    model: Optional[torch.nn.Module] = None,
    device: str = 'cuda',
    img_size: int = 512,
    subsample: int = 8
) -> Tuple[float, float]:
    """
    Compute feature and geometric matching scores for a pair of images.
    Logic replicated from mast3r_sequence_matching.ipynb.
    
    Returns:
        (feat_score, geom_score)
    """
    if model is None:
        model = MASt3RModel.get_instance(device=device)

    # Load images
    imgs = load_images([img1_path, img2_path], size=img_size, verbose=False)
    
    # Make pairs
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    
    # Inference
    with torch.no_grad():
        out = inference(pairs, model, device, batch_size=1, verbose=False)
    
    # Extract predictions
    # out keys are 'view1', 'view2', 'pred1', 'pred2'.
    # keys in pred1 are (0, 1) and (1, 0) since symmetrize=True.
    # Notebook logic iterates idx1, idx2 and gets pair (idx1, idx2).
    # Since we loaded [img1, img2], indices are 0 and 1.
    k = (0, 1)
    if k not in out['pred1']:
        # Should not happen with symmetrize=True/complete
        return 0.0, 0.0
            
    # Notebook extracts:
    # pts3d_1 = to_numpy_safe(out["pred1"]["pts3d"])[0]
    # pts3d_2 = to_numpy_safe(out["pred2"]["pts3d_in_other_view"])[0] 
    # Wait, in notebook the pair was constructed from a list of frames.
    # Here 'out' structure for pair (0,1):
    # 'pred1' is for img0, 'pred2' is for img1 (aligned to img0?)
    # Actually 'pred2' contains predictions for view2. 
    # 'pts3d_in_other_view' is specific key?
    # Let's trust the notebook key "pts3d_in_other_view" exists in pred2.
    
    # Also handle batch dimension [0]
    pred1_k = out['pred1'][k]
    pred2_k = out['pred2'][k] # This key might differ if dust3r version changed?
    
    # Check if 'pts3d_in_other_view' exists, else fallback to 'pts3d'
    # But usually 'pts3d' in pred2 is in view2 coords. 'pts3d_in_other_view' might be aligned.
    # If using AsymmetricMASt3R, it likely outputs it.
    
    pts3d_1 = to_numpy_safe(pred1_k['pts3d'])[0]
    if 'pts3d_in_other_view' in pred2_k:
        pts3d_2 = to_numpy_safe(pred2_k['pts3d_in_other_view'])[0]
    else:
        # Fallback if key missing (though notebook used it)
        pts3d_2 = to_numpy_safe(pred2_k['pts3d'])[0]

    conf1 = to_numpy_safe(pred1_k['conf'])[0]
    conf2 = to_numpy_safe(pred2_k['conf'])[0]
    
    desc1 = to_numpy_safe(pred1_k['desc'])[0]
    desc2 = to_numpy_safe(pred2_k['desc'])[0]
    
    # Feature matching
    # Notebook:
    # feat_xy1, feat_xy2, feat_conf = extract_correspondences_nonsym(
    #     torch.from_numpy(desc1), torch.from_numpy(desc2),
    #     torch.from_numpy(conf1), torch.from_numpy(conf2)
    # )
    
    feat_xy1, feat_xy2, feat_conf = extract_correspondences_nonsym(
        torch.from_numpy(desc1), torch.from_numpy(desc2),
        torch.from_numpy(conf1), torch.from_numpy(conf2)
    )
    
    # Geometric matching
    # Notebook:
    # geom_xy1, geom_xy2 = fast_reciprocal_NNs(
    #     torch.from_numpy(pts3d_1),
    #     torch.from_numpy(pts3d_2),
    #     subsample_or_initxy1=subsample
    # )
    
    geom_xy1, geom_xy2 = fast_reciprocal_NNs(
        torch.from_numpy(pts3d_1),
        torch.from_numpy(pts3d_2),
        subsample_or_initxy1=subsample
    )
    
    # Scores
    # Notebook: total_grid_points = (img_size // subsample) * (img_size // subsample) * 2
    # Verify H, W from actual output to be safe?
    H, W = pts3d_1.shape[:2]
    total_grid_points = (H // subsample) * (W // subsample) * 2
    
    if total_grid_points <= 0:
        return 0.0, 0.0
        
    feat_score = len(feat_xy1) / total_grid_points
    geom_score = len(geom_xy1) / total_grid_points
    
    return float(feat_score), float(geom_score)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_matching_scores.py img1 img2 [device]")
        sys.exit(1)
        
    img1 = sys.argv[1]
    img2 = sys.argv[2]
    dev = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    
    try:
        s1, s2 = compute_matching_scores(img1, img2, device=dev)
        print(f"Feature Score: {s1}")
        print(f"Geometric Score: {s2}")
    except Exception as e:
        print(f"Error computing scores: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
