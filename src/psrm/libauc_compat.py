"""Import the official LibAUC sampler without modifying installed modules."""
def import_official_dual_sampler():
    from libauc.sampler import DualSampler
    return DualSampler
