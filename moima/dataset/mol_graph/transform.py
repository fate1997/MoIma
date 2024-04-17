import torch
import random
from itertools import product, permutations
from copy import deepcopy


from faenet.frame_averaging import (
    frame_averaging_2D,
    frame_averaging_3D,
    data_augmentation,
)


class Transform:
    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        name = self.__class__.__name__
        items = [
            f"{k}={v}"
            for k, v in self.__dict__.items()
            if not callable(v) and k != "inactive"
        ]
        s = f"{name}({', '.join(items)})"
        if self.inactive:
            s = f"[inactive] {s}"
        return s


class FrameAveraging(Transform):
    r"""Frame Averaging (FA) Transform for (PyG) Data objects (e.g. 3D atomic graphs).

    Computes new atomic positions (`fa_pos`) for all datapoints, as well as
    new unit cells (`fa_cell`) attributes for crystal structures, when applicable.
    The rotation matrix (`fa_rot`) used for the frame averaging is also stored. 

    Args:
        frame_averaging (str): Transform method used. 
            Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
            (`"2D"`, `"3D"`, `"DA"`, `""`)
        fa_method (str): the actual frame averaging technique used.
            "stochastic" refers to sampling one frame at random (at each epoch), 
            "det" to chosing deterministically one frame, and "all" to using all frames.
            The prefix "se3-" refers to the SE(3) equivariant version of the method.
            "" means that no frame averaging is used.
            (`""`, `"stochastic"`, `"all"`, `"det"`, `"se3-stochastic"`, `"se3-all"`, `"se3-det"`)
    
    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform. 
    """

    def __init__(self, frame_averaging=None, fa_method=None):
        self.fa_method = (
            "stochastic" if (fa_method is None or fa_method == "") else fa_method
        )
        self.frame_averaging = "" if frame_averaging is None else frame_averaging
        self.inactive = not self.frame_averaging
        assert self.fa_method in {
            "",
            "stochastic",
            "det",
            "all",
            "se3-stochastic",
            "se3-det",
            "se3-all",
        }

        if self.frame_averaging:
            if self.frame_averaging == "2D":
                self.fa_func = frame_averaging_2D
            elif self.frame_averaging == "3D":
                self.fa_func = frame_averaging_3D
            elif self.frame_averaging == "DA":
                self.fa_func = data_augmentation
            elif self.frame_averaging == "svd":
                self.fa_func = frame_averaging_svd
            elif self.frame_averaging == "permute":
                self.fa_func = frame_averaging_permute
            else:
                raise ValueError(f"Unknown frame averaging: {self.frame_averaging}")

    def __call__(self, data):
        """ The only requirement for the data is to have a `pos` attribute.
        """
        if self.inactive:
            return data
        elif self.frame_averaging == "DA":
            return self.fa_func(data, self.fa_method)
        else:
            data.fa_pos, data.fa_cell, data.fa_rot = self.fa_func(
                data.pos, data.cell if hasattr(data, "cell") else None, self.fa_method
            )
            return data


def get_transform(name: str, params: dict = {}):
    if name == "frame_averaging":
        return FrameAveraging(frame_averaging='3D', fa_method="stochastic")
    elif name == "frame_averaging_permute":
        return FrameAveraging(frame_averaging='permute', fa_method="stochastic")
    elif name == "frame_averaging_svd":
        return FrameAveraging(frame_averaging='svd', fa_method="stochastic")
    else:
        raise ValueError(f"Unknown transform name: {name}")


def frame_averaging_svd(pos, cell=None, fa_method="stochastic", check=False):

    # Compute centroid and covariance
    pos = pos - pos.mean(dim=0, keepdim=True)
    u, s, v = torch.linalg.svd(pos)
    eigenval, eigenvec = s, v.T

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Compute fa_pos
    fa_pos, fa_cell, fa_rot = compute_frames_v2(eigenvec, pos, cell, fa_method, consider_perm=True)

    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def frame_averaging_permute(pos, cell=None, fa_method="stochastic", check=False):
    """Computes new positions for the graph atoms using
    frame averaging, which itself builds on the PCA of atom positions.
    Base case for 3D inputs.

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph. None if no pbc.
        fa_method (str): FA method used
            (stochastic, det, all, se3-all, se3-det, se3-stochastic)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        (tensor): updated atom positions
        (tensor): updated unit cell
        (tensor): the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos = pos - pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Compute fa_pos
    fa_pos, fa_cell, fa_rot = compute_frames_v2(eigenvec, pos, cell, fa_method, consider_perm=True)

    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def compute_frames_v2(
    eigenvec, pos, cell, fa_method="stochastic", pos_3D=None, det_index=0, consider_perm=False
):
    """Compute all `frames` for a given graph, i.e. all possible
    canonical representations of the 3D graph (of all euclidean transformations).

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): centered position vector
        cell (tensor): cell direction (dxd)
        fa_method (str): the Frame Averaging (FA) inspired technique
            chosen to select frames: stochastic-FA (stochastic), deterministic-FA (det),
            Full-FA (all) or SE(3)-FA (se3).
        pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

    Returns:
        (list): 3D position tensors of projected representation
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    permute_list = list(permutations(range(dim)))
    permute_list = [torch.LongTensor(x) for x in permute_list]
    if not consider_perm:
        permute_list = [torch.LongTensor([0, 1, 2])]
    all_fa_pos = []
    all_cell = []
    all_rots = []
    assert fa_method in {
        "all",
        "stochastic",
        "det",
        "se3-all",
        "se3-stochastic",
        "se3-det",
    }
    se3 = fa_method in {
        "se3-all",
        "se3-stochastic",
        "se3-det",
    }
    fa_cell = deepcopy(cell)
    # return [pos @ eigenvec], [fa_cell @ eigenvec], [eigenvec.unsqueeze(0)]
    if fa_method == "det" or fa_method == "se3-det":
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]
    return [pos @ eigenvec], [fa_cell], [eigenvec.unsqueeze(0)]
    for perm in permute_list:
        for pm in plus_minus_list:
            # Append new graph positions to list
            new_eigenvec = pm * eigenvec[perm, :]

            # Consider frame if it passes above check
            fa_pos = pos @ new_eigenvec

            if pos_3D is not None:
                full_eigenvec = torch.eye(3)
                fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
                full_eigenvec[:2, :2] = new_eigenvec
                new_eigenvec = full_eigenvec

            if cell is not None:
                fa_cell = cell @ new_eigenvec

            # Check if determinant is 1 for SE(3) case
            if se3 and not torch.allclose(
                torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
            ):
                continue

            all_fa_pos.append(fa_pos)
            all_cell.append(fa_cell)
            all_rots.append(new_eigenvec.unsqueeze(0))

    # Handle rare case where no R is positive orthogonal
    if all_fa_pos == []:
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))
    # Return frame(s) depending on method fa_method
    if fa_method == "all" or fa_method == "se3-all":
        return all_fa_pos, all_cell, all_rots

    elif fa_method == "det" or fa_method == "se3-det":
        return [all_fa_pos[det_index]], [all_cell[det_index]], [all_rots[det_index]]
    
    index = random.randint(0, len(all_fa_pos) - 1)
    return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]


if __name__ == '__main__':
    eigenvec = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pos = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cell = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    compute_frames_v2(eigenvec, pos, cell, fa_method="se3-all")