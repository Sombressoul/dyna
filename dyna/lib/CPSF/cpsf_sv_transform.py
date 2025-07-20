import torch

# Global spectrum -> Local vector
def spectrum_to_vector(
    spectra: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    vec_len: int,
) -> torch.Tensor:
    B, L = spectra.shape
    K = vec_len

    k = torch.arange(L, device=spectra.device).view(1, 1, L)
    n = torch.arange(K, device=spectra.device).view(1, K, 1)

    x = (scale.unsqueeze(-1) * (n / K - 0.5) + shift.unsqueeze(-1)) % 1
    phase = 2 * torch.pi * k * x
    basis = torch.exp(1j * phase)

    vectors = torch.matmul(basis, spectra.unsqueeze(-1)).squeeze(-1)
    return vectors

# Local vector -> Reconstructed spectrum
def vector_to_spectrum(
    samples: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_modes: int,
) -> torch.Tensor:
    B, K = samples.shape
    L = num_modes

    k = torch.arange(L, device=samples.device).view(1, 1, L)
    n = torch.arange(K, device=samples.device).view(1, K, 1)

    x = (scale.unsqueeze(-1) * (n / K - 0.5) + shift.unsqueeze(-1)) % 1
    phase = 2 * torch.pi * k * x
    basis = torch.exp(1j * phase)

    spectra = torch.linalg.lstsq(basis, samples.unsqueeze(-1)).solution.squeeze(-1)
    return spectra

# VERIFICATION. DO NOT TOUCH. DO NOT CHANGE.
def test(
    K: int,
    L: int,
    batch_size: int = 1,
    window_shift: float = 1.0e-3,
    max_MSE_spectra: float = 1.0e-2,
    max_MSE_vector: float = 1.0e-2,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> bool:
    """
    Task: Coordinate-Invariant Spectral Reconstruction via Toroidal Fourier Basis

    Given:
    - A complex-valued discrete global spectrum defined on a fixed global Fourier basis.
    - Arbitrary real-valued vectors, scale and shift, defining a local observation window.
    - A vector length specifying how many evenly spaced samples are extracted from the observation window.
    - A fixed number of spectral modes (Fourier components) of the global spectrum.
    - The number of spectral modes and the vector length are independent variables.
    - The number of spectral modes may be greater or less than the sample vector length.

    Objective:
    To implement two strictly defined functions:
    - spectrum_to_vector: synthesizes a vector (projected through an observation window) from a global spectrum using the specified scale and shift.
    - vector_to_spectrum: reconstructs a global spectrum from an observation vector using the same scale and shift.

    These functions must satisfy the following conditions:
    - Fully reversible: converting a spectrum to a vector and back must recover the original spectrum.
    - Coordinate-invariant: recovering the spectrum from differently scaled and shifted windows must yield the same original spectrum.
    - Fourier-consistent: the spectral representation must remain in the original fixed Fourier basis.
    - Topologically correct: all observation window coordinates must lie on a multi-dimensional torus (i.e., periodic modulo 1).
    - Fully complex: all operations, inputs, and outputs must reside in the field of complex numbers.
    - Analytically defined: no learning, fitting, or external state is allowed; all results must follow from closed-form analytic computation.
    - Non-parametric: no learnable or persistent parameters are allowed; intermediate variables may be used only for computation.

    Required verification:
    The implementation must pass a strict test consisting of:
    - Synthesizing two observation vectors from the same global (input) spectrum using different windows defined by scale and shift values.
    - Reconstructing global spectra from each observation vector using its respective scale and shift.
    - Comparing the reconstructed spectra against the original, and against each other.
    - All comparisons must yield very low mean squared error (MSE) — below machine precision thresholds.

    Allowed:
    - Approximated solution.

    Passing this test constitutes the formal definition of correctness.
    """
    max_MSE_spectra = max_MSE_spectra
    max_MSE_vector = max_MSE_vector

    samples_len = K
    num_modes = L

    # Grounding of random spectra.
    spectra_source = torch.complex(torch.randn([batch_size, num_modes]), torch.randn([batch_size, num_modes])).to(device=device)

    samples_a_scales = torch.randn([batch_size, samples_len], device=device) + 1.0
    samples_a_shifts = torch.randn([batch_size, samples_len], device=device)
    samples_b_scales = torch.randn([batch_size, samples_len], device=device) + 1.0
    samples_b_shifts = torch.randn([batch_size, samples_len], device=device)

    samples_a = spectrum_to_vector(spectra_source, samples_a_scales, samples_a_shifts, samples_len)
    samples_b = spectrum_to_vector(spectra_source, samples_b_scales, samples_b_shifts, samples_len)

    assert torch.all(samples_a.ne(samples_b)), "Samples should not be equal."
    assert samples_a.shape[-1] == samples_len, "Samples length should be fixed as defined."
    assert samples_b.shape[-1] == samples_len, "Samples length should be fixed as defined."

    spectra_a = vector_to_spectrum(samples_a, samples_a_scales, samples_a_shifts, num_modes)
    spectra_b = vector_to_spectrum(samples_b, samples_b_scales, samples_b_shifts, num_modes)

    samples_a_restored = spectrum_to_vector(spectra_a, samples_a_scales, samples_a_shifts, samples_len)
    samples_b_restored = spectrum_to_vector(spectra_b, samples_b_scales, samples_b_shifts, samples_len)
    samples_a_restored_shifted = spectrum_to_vector(spectra_a, samples_a_scales, samples_a_shifts + window_shift, samples_len)
    samples_b_restored_shifted = spectrum_to_vector(spectra_b, samples_b_scales, samples_b_shifts + window_shift, samples_len)

    mse_S_a = torch.mean((spectra_source - spectra_a)**2).abs()
    mse_S_b = torch.mean((spectra_source - spectra_b)**2).abs()
    mse_a_b = torch.mean((spectra_a - spectra_b)**2).abs()
    mse_SamplesA_SamplesARestored = torch.mean((samples_a - samples_a_restored)**2).abs()
    mse_SamplesB_SamplesBRestored = torch.mean((samples_b - samples_b_restored)**2).abs()
    mse_A_restored_shifted = torch.mean((samples_a_restored - samples_a_restored_shifted)**2).abs()
    mse_B_restored_shifted = torch.mean((samples_b_restored - samples_b_restored_shifted)**2).abs()
    cossim_a_b = torch.nn.functional.cosine_similarity(samples_a.abs(), samples_b.abs(), dim=-1).mean()

    print(f"For K {'>' if K > L else '<='} L: {K=}, {L=}")
    print(f"For {batch_size=}")
    print(f"For {window_shift=}")
    print(f"MSE between original spectra and spectra_a: {mse_S_a.item()}")
    print(f"MSE between original spectra and spectra_b: {mse_S_b.item()}")
    print(f"MSE between recovered spectra: {mse_a_b.item()}")
    print(f"MSE between samples A (source spectra) and samples A (restored spectra): {mse_SamplesA_SamplesARestored.item()}")
    print(f"MSE between samples B (source spectra) and samples B (restored spectra): {mse_SamplesB_SamplesBRestored.item()}")
    print(f"MSE between samples A (restored spectra) and samples A (restored spectra + shifted window): {mse_A_restored_shifted.item()}")
    print(f"MSE between samples B (restored spectra) and samples B (restored spectra + shifted window): {mse_B_restored_shifted.item()}")
    print(f"Cosine similarity between samples A and samples B: {cossim_a_b.item()}")

    if all([
            mse_S_a < max_MSE_spectra,
            mse_S_b < max_MSE_spectra,
            mse_a_b < max_MSE_spectra,
            mse_SamplesA_SamplesARestored < max_MSE_vector,
            mse_SamplesB_SamplesBRestored < max_MSE_vector,
        ]):
        return True
    else:
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CPSF: spectra-vector transformations.")
    parser.add_argument("-k", "--vector-length", help="Length of vector representation", required=False, default=1024, type=int)
    parser.add_argument("-l", "--spectral-mods", help="Number of spectral mods", required=False, default=128, type=int)
    parser.add_argument("-w", "--window-shift", help="Window shift", required=False, default=1.0e-3, type=float)
    parser.add_argument("-b", "--batch-size", help="Batch size", required=False, default=512, type=int)
    parser.add_argument("-s", "--seed", help="Random seed (default=1337, random=-1)", required=False, default=None, type=int)
    parser.add_argument("--mse-spectra", help="Restored spectra MSE criterion", required=False, default=1.0e-2, type=float)
    parser.add_argument("--mse-vector", help="Restored vector MSE criterion", required=False, default=1.0e-2, type=float)
    args = parser.parse_args()

    if args.seed is None:
        torch.manual_seed(1337)
    elif args.seed != -1:
            torch.manual_seed(args.seed)
    
    K = args.vector_length
    L = args.spectral_mods
    B = args.batch_size
    W = args.window_shift

    with torch.no_grad():
        K_to_L = test(K, L, B, W, args.mse_spectra, args.mse_vector)
        print("\n")
        L_to_K = test(L, K, B, W, args.mse_spectra, args.mse_vector)
        print("\n")

    print(f"For K({K}) to L({L}) at B={B}, MSE criteria S/V = {args.mse_spectra}/{args.mse_vector}: {K_to_L}")
    print(f"For L({L}) to K({K}) at B={B}, MSE criteria S/V = {args.mse_spectra}/{args.mse_vector}: {L_to_K}")

    if K_to_L and L_to_K:
        print("SUCCESS")
    else:
        print("FAIL")
