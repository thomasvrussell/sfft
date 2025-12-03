import numpy as np
# version: Dec 2, 2025
__author__ = "Lei Hu <leihu@sas.upenn.edu>"

def KERNEL_CSZ(KERNEL, NX_IMG, NY_IMG, NORMALIZE_KERNEL=False):
    """ Circular Shift the kernel and extend to the target size """
    N0, N1 = NX_IMG, NY_IMG
    L0, L1 = KERNEL.shape
    W0, W1 = (L0 - 1)//2, (L1 - 1)//2
    # Note: currently only support odd-sized kernel
    assert L0 % 2 == 1 and L1 % 2 == 1   
    
    if NORMALIZE_KERNEL:
        KERNEL_TZP = np.pad(KERNEL / np.sum(KERNEL), \
            pad_width=((0, N0 - L0), (0, N1 - L1)), mode='constant', constant_values=0.)
    else:   
        KERNEL_TZP = np.pad(KERNEL, pad_width=((0, N0 - L0), (0, N1 - L1)), \
            mode='constant', constant_values=0.)
    KIMG_CSZ = np.roll(np.roll(KERNEL_TZP, -W0, axis=0), -W1, axis=1)
    return KIMG_CSZ

def KERNEL_CSZ_INV(KIMG, NX_KERN, NY_KERN, VERBOSE_LEVEL=2):
    """ Inverse Circular Shift the kernel and truncate to the target size """
    L0, L1 = NX_KERN, NY_KERN
    W0, W1 = (L0 - 1)//2, (L1 - 1)//2
    # Note: currently only support odd-sized kernel
    assert L0 % 2 == 1 and L1 % 2 == 1
    
    KIMG_iCSZ = np.roll(np.roll(KIMG, W1, axis=1), W0, axis=0)
    KERNEL = KIMG_iCSZ[:L0, :L1]
    if VERBOSE_LEVEL in [1, 2]:
        LOSE_RATIO = 1. - np.sum(np.abs(KERNEL)) / np.sum(np.abs(KIMG_iCSZ))
        _report_message = "Kernel Truncation Loses APE = [%.4f %s]" %(LOSE_RATIO*100, '%')
        print("MeLOn CheckPoint: %s " % _report_message)
    return KERNEL

def DeCorrelation_Calculator(NX_IMG, NY_IMG, KERNEL_JQueue, BKGSIG_JQueue, KERNEL_IQueue=[], BKGSIG_IQueue=[], 
    MATCH_KERNEL=None, REAL_OUTPUT=False, REAL_OUTPUT_SIZE=None, NORMALIZE_OUTPUT=True, VERBOSE_LEVEL=2):

    NUM_I, NUM_J = len(KERNEL_IQueue), len(KERNEL_JQueue)
    assert NUM_J > 0

    DELTA_KERNEL = np.array([
        [0, 0, 0], 
        [0, 1, 0], 
        [0, 0, 0]], dtype=np.float64
    )

    FDENO = None
    for KERNEL, BKGSIG in zip(KERNEL_JQueue, BKGSIG_JQueue):
        if KERNEL is not None:
            K_CSZ = KERNEL_CSZ(KERNEL=KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
        else:
            K_CSZ = KERNEL_CSZ(KERNEL=DELTA_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
        FK_CSZ = np.fft.fft2(K_CSZ)
        FK2_CSZ = (np.conj(FK_CSZ) * FK_CSZ).real
        if FDENO is None:
            FDENO = (BKGSIG**2 * FK2_CSZ) / NUM_J**2
        else: 
            FDENO += (BKGSIG**2 * FK2_CSZ) / NUM_J**2

    if MATCH_KERNEL is not None:
        MK_CSZ = KERNEL_CSZ(KERNEL=MATCH_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
    else:
        MK_CSZ = KERNEL_CSZ(KERNEL=DELTA_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)

    FMK_CSZ = np.fft.fft2(MK_CSZ)
    FMK2_CSZ = (np.conj(FMK_CSZ) * FMK_CSZ).real

    for KERNEL, BKGSIG in zip(KERNEL_IQueue, BKGSIG_IQueue):
        if KERNEL is not None:
            K_CSZ = KERNEL_CSZ(KERNEL=KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
        else:
            K_CSZ = KERNEL_CSZ(KERNEL=DELTA_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
        FK_CSZ = np.fft.fft2(K_CSZ)
        FK2_CSZ = (np.conj(FK_CSZ) * FK_CSZ).real
        FDENO += (BKGSIG**2 * FK2_CSZ * FMK2_CSZ) / NUM_I**2

    FDENO = np.sqrt(FDENO)
    FKDECO = 1. / FDENO

    if not REAL_OUTPUT:
        if NORMALIZE_OUTPUT:
            NORMALIZE_FACTOR = 1./FKDECO[0, 0]
            FKDECO *= NORMALIZE_FACTOR
        return FKDECO

    if REAL_OUTPUT:
        KDECO = np.fft.ifft2(FKDECO).real
        KDECO = KERNEL_CSZ_INV(
            KIMG=KDECO, NX_KERN=REAL_OUTPUT_SIZE[0], NY_KERN=REAL_OUTPUT_SIZE[1],
            VERBOSE_LEVEL=VERBOSE_LEVEL
        )
        
        if NORMALIZE_OUTPUT:
            assert REAL_OUTPUT_SIZE is not None
            NORMALIZE_FACTOR = 1./np.sum(KDECO)
            KDECO *= NORMALIZE_FACTOR
        return KDECO
