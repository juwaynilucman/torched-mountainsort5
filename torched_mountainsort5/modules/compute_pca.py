import torch
import torch.nn as nn

from ..schema import SortingBatch, SortingParameters


class ComputePCA(nn.Module):
    """Reduce snippet dimensionality via PCA (torch-native SVD solvers).

    Reads:  batch.snippets
    Writes: batch.features
    """

    def __init__(self, params: SortingParameters):
        super().__init__()
        self.npca_per_channel = params.npca_per_channel

    def forward(self, batch: SortingBatch) -> SortingBatch:
        snippets = batch.snippets
        assert snippets is not None

        L, T, M = snippets.shape
        npca = self.npca_per_channel * M
        flat = snippets.reshape(L, T * M)

        batch.features = _compute_pca_features(flat, npca=npca)
        return batch


def _pick_solver(n_samples: int, n_features: int, n_components: int) -> str:
    if n_features <= 1000 and n_samples >= 10 * n_features:
        return "covariance_eigh"
    if max(n_samples, n_features) > 500 and n_components < 0.8 * min(n_samples, n_features):
        return "randomized"
    return "full"


def _svd_flip_u_based(U: torch.Tensor, Vt: torch.Tensor):
    idx = torch.argmax(torch.abs(U), dim=0)
    signs = torch.sign(U[idx, torch.arange(U.shape[1], device=U.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    U = U * signs
    Vt = Vt * signs[:, None]
    return U, Vt


def _compute_pca_features(
    X: torch.Tensor,
    *,
    npca: int,
    random_state: int = 0,
) -> torch.Tensor:
    L, D = X.shape
    k = min(npca, L, D)
    out_dtype = torch.float32 if X.dtype not in (torch.float32, torch.float64) else X.dtype

    if L == 0 or D == 0:
        return torch.zeros((L, k), dtype=out_dtype, device=X.device)
    if k == 0:
        return torch.zeros((L, 0), dtype=out_dtype, device=X.device)

    Xw = X.to(out_dtype)
    mean = Xw.mean(dim=0, keepdim=True)
    Xc = Xw - mean

    solver = _pick_solver(L, D, k)

    if solver == "covariance_eigh":
        C = (Xc.mT @ Xc) / max(L - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvecs = torch.flip(eigvecs, dims=[1])
        Vt = eigvecs.mT[:k]
        scores = Xc @ Vt.mT
        return scores.to(torch.float32)

    if solver == "randomized":
        q = min(k + 10, min(L, D))
        if X.is_cuda:
            torch.cuda.manual_seed_all(random_state)
        torch.manual_seed(random_state)

        U, S, V = torch.pca_lowrank(Xc, q=q, center=False, niter=4)
        U = U[:, :k]
        S = S[:k]

        idx = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[idx, torch.arange(k, device=U.device)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        U = U * signs

        return (U * S).to(torch.float32)

    # full SVD fallback
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    U, Vt = _svd_flip_u_based(U, Vt)
    return (U[:, :k] * S[:k]).to(torch.float32)
