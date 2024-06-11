from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train(config, generator, discriminator, kp_detector, he_estimator, checkpoint, log_dir, dataset, device_ids, he_estimator_far=None):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_he_estimator = torch.optim.Adam(he_estimator.parameters(), lr=train_params['lr_he_estimator'], betas=(0.5, 0.999))


    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, he_estimator,
                                      optimizer_generator, optimizer_discriminator, optimizer_kp_detector, optimizer_he_estimator)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_he_estimator = MultiStepLR(optimizer_he_estimator, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
                                        
    # deepcopy of he_estimator, optimizer_he_estimator, scheduler_he_estimator
    import copy
    he_estimator_far = copy.deepcopy(he_estimator)
    optimizer_he_estimator_far = copy.deepcopy(optimizer_he_estimator)
    scheduler_he_estimator_far = copy.deepcopy(scheduler_he_estimator)



    is_freeze_he = config['train_params']['is_freeze_he']


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, he_estimator, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'], he_estimator_far=he_estimator_far)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)


    if is_freeze_he:
        # freeze generator_full.he_estimator
        for param in generator_full.he_estimator.parameters():
            param.requires_grad = False 
        generator_full.he_estimator.eval()
        # freeze he_estimator
        for param in he_estimator.parameters():
            param.requires_grad = False
        he_estimator.eval()
   

   
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)



    import tqdm
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            print('epoch: ', epoch)
            # tqdm
            for x in tqdm.tqdm(dataloader):
                losses_generator, generated = generator_full(x)
                # # check he_estimator_far's weight is equal to he_estimator 
                # for param, param_far in zip(he_estimator.parameters(), he_estimator_far.parameters()):
                #     assert torch.all(torch.eq(param, param_far))                


                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                if not is_freeze_he:
                    optimizer_he_estimator.step()
                    optimizer_he_estimator.zero_grad()
                optimizer_he_estimator_far.step()
                optimizer_he_estimator_far.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            if not is_freeze_he:
                scheduler_he_estimator.step()
            scheduler_he_estimator_far.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'he_estimator': he_estimator,
                                     'he_estimator_far': he_estimator_far,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_he_estimator': optimizer_he_estimator,
                                     'optimizer_he_estimator_far': optimizer_he_estimator_far}, inp=x, out=generated)