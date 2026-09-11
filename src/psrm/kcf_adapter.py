"""Mechanical KCF box-to-three-plane projection; no labels or reliability rules."""
from pathlib import Path
import sys
import numpy as np

from psrm.crep_canonicalizer import (
    _global_sampling_grid, _bilinear_sample_zero, _fractional_box_occupancy,
)

def project(response, roi, resized, candidate_box, image_size, resolution=32):
    """Map displacement-grid evidence to image-coordinate box support.

    KCF response cell (u,v) corresponds, before clipping/rounding, to box
    center s*(roi.x+u+1, roi.y+v+1), s in {1,2}. The *unchanged output box*
    is mapped back to this grid; its support may differ from the raw peak
    after native clipping. Signed response values are divided by max(abs).
    """
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2 or not np.isfinite(response).all():
        raise ValueError('a finite two-dimensional KCF response is required')
    rh,rw = response.shape
    iw,ih = map(int,image_size)
    x,y,w,h = map(float,candidate_box)
    factor = 2.0 if resized else 1.0
    ox,oy = factor*(float(roi[0])+1),factor*(float(roi[1])+1)
    magnitude = float(np.abs(response).max())
    task = response / magnitude if magnitude > 0 else np.zeros_like(response)
    gx,gy,scale,left,top = _global_sampling_grid(rh,rw,resolution)
    task_plane = _bilinear_sample_zero(task,gx,gy)
    yy,xx = np.indices((rh,rw),dtype=np.float64)
    image_x,image_y = ox+factor*xx,oy+factor*yy
    # Valid displacement hypotheses have centers in the image. Letterbox
    # padding is separately zeroed by the existing sampling helper.
    valid = ((image_x>=0)&(image_x<iw)&(image_y>=0)&(image_y<ih)).astype(np.float32)
    valid_plane = _bilinear_sample_zero(valid,gx,gy)
    cx,cy = x+w/2,y+h/2
    in_frame = w>0 and h>0 and 0<=cx<iw and 0<=cy<ih
    support = np.zeros((resolution,resolution),dtype=np.float32)
    boundary = False
    if w>0 and h>0:
        # Add 0.5 to convert field cell centers to field cell edges.
        edges = ((x-ox)/factor+0.5,(y-oy)/factor+0.5,
                 (x+w-ox)/factor+0.5,(y+h-oy)/factor+0.5)
        boundary = edges[0]<0 or edges[1]<0 or edges[2]>rw or edges[3]>rh
        projected = (left+scale*edges[0],top+scale*edges[1],
                     left+scale*edges[2],top+scale*edges[3])
        support = _fractional_box_occupancy((resolution,resolution),projected)
    dense = np.stack([task_plane,support,valid_plane]).astype(np.float32)
    metadata = np.asarray([cx/iw,cy/ih,float(in_frame),float(boundary)],np.float32)
    provenance = {'response_scale':magnitude,'image_origin_x':ox,'image_origin_y':oy,
                  'image_step':factor,'field_to_grid_scale':scale}
    return dense,metadata,provenance
