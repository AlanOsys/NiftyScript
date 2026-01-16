def reconstructionScript(
        datain, mumaps, hst, scanner_params, recmod=3, itr=4, fwhm=0., psf=None,
        mask_radius=29., decay_ref_time=None, attnsino=None, sctsino=None, randsino=None,
        normcomp=None, emmskS=False, frmno='', fcomment='', outpath=None, fout=None,
        store_img=False, store_itr=None, ret_sinos=False):
    muh, muo = mumaps
    # get the GPU version of the image dims
    log.info("1")
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    ######

    ######
    Cnt = scanner_params['Cnt']
    mus = convert2dev(muo + muh, Cnt)
    #psng = remgaps(hst['psino'], txLUT, Cnt)
    log.info("cu check")
    #asng = cu.zeros(psng.shape, dtype=np.float32)
    log.info("2")
    petprj.fprj(mus, cu.asarray(mumaps), txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt, 1)

    #ncmp, _ = get_components(datain, Cnt)

    #nsng = get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=True)
    #ansng = asng*nsng
    #ISUB_DEFAULT = np.array([-1], dtype=np.int32)
    #tmpsens = cu.zeros((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
    #log.info("3")
    #petprj.bprj(tmpsens, cu.asarray(ansng), txLUT, axLUT, ISUB_DEFAULT,Cnt)
    return Cnt
