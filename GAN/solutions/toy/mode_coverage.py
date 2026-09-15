def mode_coverage(samples, centers=CENTERS, tol=3.0, sigma=SIGMA_RING, min_share=0.01):
    """Two numbers that a GAN loss will never give you.

    `covered` counts the modes that receive at least `min_share` of the samples,
    a sample being assigned to the nearest centre and kept only if it falls within
    `tol` standard deviations of it. `quality` is the share of samples that land
    near *some* mode.

    The pair matters more than either number: a generator that puts everything on a
    single mode has a quality close to 1 and a coverage of 1 out of 8. Quality alone
    would call it excellent.
    """
    samples = samples.detach().cpu()
    distance = torch.cdist(samples, centers.cpu())
    nearest, index = distance.min(dim=1)

    close = nearest < tol * sigma
    counts = torch.bincount(index[close], minlength=len(centers))
    covered = int((counts > min_share * len(samples)).sum())
    quality = close.float().mean().item()

    return covered, quality, counts.tolist()


covered, quality, counts = mode_coverage(fake_samples)
print(f"modes covered      : {covered} / {len(CENTERS)}")
print(f"samples near a mode: {quality:.1%}")
print(f"samples per mode   : {counts}")
