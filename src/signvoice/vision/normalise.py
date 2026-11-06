"""
normalise.py - transform raw MediaPipe landmark features into model-ready vectors.

This module provides the geometric transformations used to make landmark
features more consistent across signers, camera positions, and recording
conditions. It operates on the flattened feature layout defined in
signvoice.vision.landmark: 147 values representing 49 three-dimensional points
(21 left-hand points, 21 right-hand points, and 7 selected pose points).

The main entry point is normalise(). It first converts the input to float32 and
optionally adjusts the horizontal coordinate according to the source image's
aspect ratio. It then obtains a reference frame, centres the landmarks relative
to that reference, and scales the centred coordinates by a reference distance.
The default body reference uses the midpoint between the shoulders as the
origin and shoulder width as the scale unit. The hand reference instead centres
each hand on its wrist and scales it by palm length; pose points are not used
with this reference.

The module also exposes the individual operations so they can be used
separately:
    - as_points() and the internal conversion helpers reshape between flattened
      feature vectors and arrays of (x, y, z) landmark coordinates.
    - body_reference() and hand_reference() construct the origins, scale
      factors, and validity masks used by the centring and scaling operations.
    - centeralize() removes the selected reference origin and sets landmarks
      without a valid reference to zero.
    - scale() divides centred coordinates by the corresponding reference
      distance.
    - mirror() swaps left/right landmark identities and negates the x
      coordinates to produce a horizontally mirrored representation.
    - aspect_from_size() derives an image width-to-height ratio from metadata.

Inputs may contain a single frame with shape (FEATURE_DIM,) or batches/sequences
with arbitrary leading dimensions and a final feature dimension of FEATURE_DIM.
The returned feature vectors retain the same leading dimensions and flattened
feature layout.

The body reference is intended for dynamic signs where hand position relative
to the torso matters. The hand reference is intended for static handshape
classification where translation and overall hand size should be removed.
"""

from __future__ import annotations
 
from dataclasses import dataclass
from typing import Callable
 
import numpy as np
 
from signvoice.vision.landmark import FEATURE_DIM, LEFT_HAND, ONE_HAND_PTS, POSE, POSE_POS, RIGHT_HAND
 
# The 147 values viewed as 49 points of (x, y, z): left hand, right hand, then the pose joints.
N_POINTS = FEATURE_DIM // 3
LEFT_START, RIGHT_START, POSE_START = LEFT_HAND.start // 3, RIGHT_HAND.start // 3, POSE.start // 3 #Integers representing the starting landmark index.
L_SHOULDER = POSE_START + POSE_POS["l_shoulder"]
R_SHOULDER = POSE_START + POSE_POS["r_shoulder"]
WRIST, MIDDLE_MCP = 0, 9           # wrist and the base of the middle finger indices inside MediaPipe's 21-point hand model.
EPS = 1e-6            

# UTILITY FUNCTIONS

def _as_float32(features: np.ndarray) -> np.ndarray: 
    """
    Convert landmark features to float32 and validate their feature dimension.

    Args:
        features(np.ndarray): Flattened landmark feature array. Its final
            dimension must contain FEATURE_DIM values.

    Returns:
        np.ndarray: The input represented as a float32 NumPy array.

    Raises:
        ValueError: If the final dimension does not match FEATURE_DIM.
    """    
    
    features = np.asarray(features, dtype=np.float32)
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected last dimension {FEATURE_DIM}, got array of shape {features.shape}")
    return features

def as_points(features: np.ndarray) -> np.ndarray:
    """
    Reshape flattened landmark features into individual 3D landmark points.

    Args:
        features(np.ndarray): Feature array with shape (..., FEATURE_DIM).

    Returns:
        np.ndarray: Landmark coordinates with shape (..., N_POINTS, 3),
            where each point contains its x, y, and z coordinates.

    Raises:
        ValueError: If the final feature dimension is not FEATURE_DIM.
    """    
    
    features = _as_float32(features)
    return features.reshape(*features.shape[:-1], N_POINTS, 3)

def _flatten(pts: np.ndarray) -> np.ndarray:
    """
    Flatten a structured landmark array back into the feature-vector layout.

    Args:
        pts(np.ndarray): Landmark coordinates with shape (..., N_POINTS, 3).

    Returns:
        np.ndarray: Float32 feature array with shape (..., FEATURE_DIM).
    """
        
    return pts.reshape(*pts.shape[:-2], FEATURE_DIM).astype(np.float32, copy= False)

def detected(pts: np.ndarray) -> np.ndarray:
    """
    Identify landmark points that contain a non-zero coordinate.

    A point is considered detected when at least one of its x, y, or z
    coordinates is non-zero. This follows the zero-filled missing-landmark
    representation used by the landmark extraction pipeline.

    Args:
        pts(np.ndarray): Landmark coordinates with shape (..., N_POINTS, 3).

    Returns:
        np.ndarray: Boolean detection mask with shape (..., N_POINTS).
    """

    return np.any(pts != 0, axis=-1)

# REFERNCING OBJECTS

@dataclass(frozen= True)
class Reference:
    """
    Store the geometric reference values used to normalize landmark points.

    The arrays are broadcast across the landmark dimension so each point can
    use the appropriate origin, scale factor, and validity flag during
    centring and scaling.

    Attributes:
        origin(np.ndarray): Reference coordinates with shape (..., N_POINTS, 3).
            Each landmark is centred relative to its corresponding origin.
        scale(np.ndarray): Positive scale factors with shape (..., N_POINTS).
            Each landmark's coordinates are divided by its corresponding value.
        valid(np.ndarray): Boolean mask with shape (..., N_POINTS), indicating
            whether a usable reference was available for each landmark.
    """
        
    origin: np.ndarray       # (..., 49, 3) the point each landmark is measured from
    scale: np.ndarray        # (..., 49)    the length each landmark is divided by
    valid: np.ndarray        # (..., 49)    False = no usable reference, the point becomes zero
    
def body_reference(features: np.ndarray) -> Reference:
    """
    Build a body-relative reference using the two shoulder landmarks.

    The midpoint between the left and right shoulders becomes the frame origin.
    The Euclidean distance between the shoulders in the x-y plane becomes the
    scale unit. The origin's z coordinate is set to zero, so z is scaled by
    shoulder width but is not translated relative to the shoulder midpoint.

    If either shoulder is missing or the shoulder width is too small, the
    reference is marked invalid and a scale of 1.0 is used to avoid division
    by zero. This reference preserves the hands' positions relative to the
    upper body, which is useful for dynamic sign classification.

    Args:
        features(np.ndarray): Raw landmark features with shape
            (..., FEATURE_DIM).

    Returns:
        Reference: Broadcast reference origins, shoulder-width scale factors,
            and validity mask for all landmark points.
    """
    
    pts = as_points(features)
    left, right = pts[..., L_SHOULDER, :], pts[..., R_SHOULDER, :]
    width = np.linalg.norm((left-right)[...,:2], axis= -1)                  # Calculates the eucadilian distance between the right and left shoulder landmarks 
    condition = np.asarray(left.any(-1) & right.any(-1) & (width > EPS))    # Creates a boolean mask for initialized x and y values
    origin = np.where(condition[..., None], (left + right)/2, 0.0)          # None is added for the z values 
    origin[..., 2] = 0.0                                                    #z is scaled by the shoulder width but NOT centred.
    unit = np.where(condition, width, 1.0)                                  #If the reference is valid, shoulder width becomes one unit. 
                                                                            #Otherwise, the unit is set to 1 to avoid division by zero.
    
    return Reference(
        origin=np.broadcast_to(origin[..., None, :], pts.shape),         #(..., 49, 3)
        scale=np.broadcast_to(unit[..., None], pts.shape[:-1]),          #(..., 49)
        valid=np.broadcast_to(condition[..., None], pts.shape[:-1]),     #(..., 49)   
    )

def hand_reference(features: np.ndarray) -> Reference:
    """
    Build independent wrist-centred references for the left and right hands.

    Each hand uses its wrist as the origin and the 3D distance from the wrist to
    the middle-finger MCP joint as its scale unit. This removes each hand's
    translation and overall size while retaining its relative handshape.
    Pose landmarks are assigned no valid reference and therefore become zero
    during centring. This reference is intended for static handshape
    classification and can be used when pose features are absent.

    A hand reference is valid only when its wrist is present and its palm
    length exceeds EPS. Invalid hands retain a scale factor of 1.0 to avoid
    division by zero.

    Args:
        features(np.ndarray): Raw landmark features with shape
            (..., FEATURE_DIM).

    Returns:
        Reference: Per-hand wrist origins, palm-length scale factors, and
            validity mask for all landmark points.
    """
    pts = as_points(features)
    origin = np.zeros_like(pts)
    unit = np.ones(pts.shape[:-1], np.float32)
    valid = np.zeros(pts.shape[:-1], bool)
    for start in (LEFT_START, RIGHT_START):
        block = slice(start, start + ONE_HAND_PTS)
        wrist, base = pts[..., start + WRIST, :], pts[..., start + MIDDLE_MCP, :]
        palm = np.linalg.norm(base - wrist, axis=-1)
        ok = np.asarray(wrist.any(-1) & (palm > EPS))
        origin[..., block, :] = np.where(ok[..., None, None], wrist[..., None, :], 0.0)
        unit[..., block] = np.where(ok[..., None], palm[..., None], 1.0)
        valid[..., block] = ok[..., None]
    return Reference(origin, unit, valid)
 
 
REFERENCES: dict[str, Callable[..., Reference]] = {"body": body_reference, "hand": hand_reference}
 
 
def get_reference(name: str) -> Callable[..., Reference]:
    """
    Retrieve a reference-construction function by its registered name.

    Args:
        name(str): Registered reference name. Currently supports "body" and
            "hand".

    Returns:
        Callable[..., Reference]: Function that constructs the requested
            Reference from a feature array.

    Raises:
        ValueError: If name is not present in REFERENCES.
    """
    
    try:
        return REFERENCES[name]
    except KeyError:
        raise ValueError(f"unknown reference {name!r}; choose from {sorted(REFERENCES)}") from None
    

def centeralize(features: np.ndarray, ref: Reference) -> np.ndarray:
    """
    Translate valid landmark points relative to their reference origins.

    A point is retained only when it is detected and its reference is valid.
    Points that fail either condition are replaced with zeros. The operation
    changes the coordinates in place conceptually but returns a new flattened
    feature array.

    Args:
        features(np.ndarray): Landmark features with shape (..., FEATURE_DIM).
        ref(Reference): Reference origins and validity mask, usually produced
            by body_reference() or hand_reference().

    Returns:
        np.ndarray: Centred float32 features with shape (..., FEATURE_DIM).
    """
        
    pts = as_points(features)
    keep = detected(features) & ref.valid
    
    return _flatten(np.where(keep[..., None], pts - ref.origin, 0.0))

def scale(features: np.ndarray, ref: Reference) -> np.ndarray:
    """
    Divide landmark coordinates by their corresponding reference distances.

    The scale factor is applied uniformly to x, y, and z for each landmark.
    For body-relative normalization this is typically shoulder width; for
    hand-relative normalization it is the corresponding hand's palm length.
    This operation is intended to receive already-centred features, so
    landmarks invalidated by centeralize() remain zero.

    Args:
        features(np.ndarray): Centred landmark features with shape
            (..., FEATURE_DIM).
        ref(Reference): Reference scale factors with shape compatible with
            the input's leading dimensions and landmark count.

    Returns:
        np.ndarray: Scaled float32 features with shape (..., FEATURE_DIM).
    """
    
    pts = as_points(features)
    return _flatten(pts/ref.scale[..., None])

def _mirroed_ordering() -> np.ndarray:
    """
    Construct the landmark-index permutation for horizontal mirroring.

    The permutation exchanges all 21 left-hand points with their corresponding
    right-hand points. It also exchanges the left/right shoulder, elbow, and
    wrist pose points. Landmark identities are swapped as complete (x, y, z)
    points; coordinate components are not independently reordered.

    Returns:
        np.ndarray: Integer index array of length N_POINTS describing the
            mirrored landmark ordering.
    """
    
    order = list(range(N_POINTS))
    order[LEFT_START:LEFT_START + ONE_HAND_PTS] = range(RIGHT_START, RIGHT_START + ONE_HAND_PTS) 
    order[RIGHT_START:RIGHT_START + ONE_HAND_PTS] = range(LEFT_START, LEFT_START + ONE_HAND_PTS)   
     
    for left, right in  (("l_shoulder", "r_shoulder"), ("l_elbow", "r_elbow"), ("l_wrist", "r_wrist")): #Creates a permutation array (an "order" or index mapping) to horizontally flip (mirror) pose estimation landmarks
        a, b = POSE_START + POSE_POS[left], POSE_START + POSE_POS[right]                                # Determines the index of the paired points to swap
        order[a], order[b] = b, a                                                                       # Swaps the values at indecies a and b 
    
    return np.ndarray(order)
    
    
MIRROR_ORDER = _mirroed_ordering() # Calculates the ordering once when the module is imported.
    
    
def mirror(features : np.ndarray) -> np.ndarray:
    """
    Horizontally mirror landmark features and exchange left/right identities.

    The landmark order is permuted using MIRROR_ORDER, then each point's x
    coordinate is negated. The y and z coordinates are retained. The input is
    not intentionally modified; the reordered point array is transformed and
    returned in the flattened feature layout.

    Args:
        features(np.ndarray): Landmark features with shape (..., FEATURE_DIM).

    Returns:
        np.ndarray: Mirrored float32 features with shape (..., FEATURE_DIM).
    """
        
    pts = as_points(features)
    out = pts[..., MIRROR_ORDER, :] # Changes the ordring of 49 in (..., 49, 3) on a copy no the orignal pts
    out[..., 0] *= -1               # Reflects or negates the x values along the last (..., 3) dim of the array

    return _flatten(out)

def _stretch_x(features: np.ndarray, aspect: float) -> np.ndarray:
    """
    Adjust horizontal landmark coordinates for the source image aspect ratio.

    When aspect differs from 1.0, x coordinates are multiplied by that ratio.
    This compensates for the difference between normalized image-space x and y
    units when the source image is not square. Zero-filled missing points
    remain zero.

    Args:
        features(np.ndarray): Landmark features with shape (..., FEATURE_DIM).
        aspect(float): Image width divided by image height.

    Returns:
        np.ndarray: Features with adjusted x coordinates, retaining the
            original flattened shape.
    """
        
    if aspect == 1.0:
        return features
    pts = as_points(features).copy()
    pts[..., 0] *= aspect                    # zeros stay zero
    return _flatten(pts)
 
 
def aspect_from_size(width, height, default: float = 1.0) -> float:
    """
    Calculate the image width-to-height ratio from metadata values.

    Width and height may be numeric values or strings, as commonly read from
    metadata.csv. Missing, blank, non-numeric, non-finite, or non-positive
    dimensions cause the supplied default to be returned. For older recordings
    known to use a fixed resolution, provide that resolution's ratio as default.

    Args:
        width(int | float | str | None): Image width from recording metadata.
        height(int | float | str | None): Image height from recording metadata.
        default(float): Fallback ratio when the dimensions cannot be used.

    Returns:
        float: width / height when both dimensions are valid; otherwise default.
    """
    
    try:
        w, h = float(width), float(height)
    except (TypeError, ValueError):
        return default
    if not (np.isfinite(w) and np.isfinite(h)) or w <= 0 or h <= 0:
        return default
    return w / h

def normalise(features: np.ndarray, reference: Callable[..., Reference] = body_reference, aspect: float = 1.0) -> np.ndarray:
    """
    Apply aspect correction, reference-based centring, and scale normalization.

    The operations are applied in this order:
        1. Convert the input to float32 and validate its feature dimension.
        2. Adjust x coordinates using the supplied image aspect ratio.
        3. Construct a reference from the adjusted features.
        4. Centre detected points using the reference origins.
        5. Divide the centred coordinates by the reference scale factors.

    Use body_reference for dynamic signs where hand position relative to the
    torso matters. Use hand_reference for static handshape classification where
    hand translation and overall size should be removed.

    Args:
        features(np.ndarray): Raw landmark feature array with shape
            (..., FEATURE_DIM), such as (FEATURE_DIM,) for one frame or
            (T, FEATURE_DIM) for a sequence.
        reference(Callable[..., Reference]): Function used to construct the
            normalization reference. Defaults to body_reference; hand_reference
            is the alternative built-in reference.
        aspect(float): Source image width-to-height ratio. Use 1.0 when the
            aspect ratio is unknown or when no horizontal correction is needed.

    Returns:
        np.ndarray: Normalized float32 feature array with the same leading
            dimensions and final dimension FEATURE_DIM.

    Raises:
        ValueError: If the input's final dimension is not FEATURE_DIM.
    """
    
    feats = _stretch_x(_as_float32(feats), aspect)
    ref = reference(feats)
    return scale(centeralize(feats, ref), ref)