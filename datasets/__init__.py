from . import cardiacspect_dataset


def get_datasets_valid(opts):
    if opts.dataset == 'CardiacSPECT':
        trainset = cardiacspect_dataset.CardiacSPECT_Train(opts)
        validset = cardiacspect_dataset.CardiacSPECT_Valid(opts)

    elif opts.dataset == 'XXX':
        a = 1
        # trainset = sv_dataset.SVTrain(opts.data_root)
        # valset = sv_dataset.SVTest(opts.data_root)

    return trainset, validset


def get_datasets_test(opts):
    if opts.dataset == 'CardiacSPECT':
        trainset = cardiacspect_dataset.CardiacSPECT_Train(opts)
        testset = cardiacspect_dataset.CardiacSPECT_Test(opts)

    elif opts.dataset == 'XXX':
        a = 1
        # trainset = sv_dataset.SVTrain(opts.data_root)
        # valset = sv_dataset.SVTest(opts.data_root)

    return trainset, testset