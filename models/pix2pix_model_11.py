import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['argo_50','G_GAN_50','argo_100','G_GAN_100','argo_150','G_GAN_150']
        #self.loss_names = ['argo_20','argo_50','argo_100','argo_150','argo_200','argo_300']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B50','fake_B100','fake_B150']
        self.min_0 = []
        self.min_20 = []
        self.min_50 = []
        self.min_100 = []
        self.min_150 = []
        self.min_200 = []
        self.min_300 = []
        self.min_00 = []
        self.min_20_0 = []
        self.min_50_0  = []
        self.min_100_0  = []
        self.min_150_0  = []
        self.min_200_0  = []
        self.min_300_0  = []
        self.max_200=[]
        self.max_200_0=[]
        self.max_200_0_0=[]
        self.min_200_0_0=[]
        self.max_200_1=[]
        self.min_200_1=[]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G','D50','D100','D150']
            #self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G','D50','D100','D150']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout,  opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_E = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
    #    if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        self.criterionL1 = torch.nn.L1Loss()
        self.netD50 = networks.define_D(opt.input_nc+ opt.output_nc ,  opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD100 = networks.define_D(opt.input_nc+ opt.output_nc ,  opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD150 = networks.define_D(opt.input_nc+ opt.output_nc ,  opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_E = torch.optim.Adam(self.netG_E.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

            self.optimizer_D50 = torch.optim.Adam(self.netD50.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D100 = torch.optim.Adam(self.netD100.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D150 = torch.optim.Adam(self.netD150.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_E)

            self.optimizers.append(self.optimizer_D50)
            self.optimizers.append(self.optimizer_D100)
            self.optimizers.append(self.optimizer_D150)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B20'].to(self.device)
        self.real_B50 = input['B50' if AtoB else 'A'].to(self.device)

        self.real_B100 = input['B100' if AtoB else 'A'].to(self.device)

        self.real_B150 = input['B150' if AtoB else 'A'].to(self.device)



        self.m0 = input['m0'].to(self.device)
        self.m50 = input['m50'].to(self.device)

        self.m100 = input['m100'].to(self.device)
        self.m150 = input['m150'].to(self.device)
        self.mask0_0 = (self.m0 != -1)

        self.mask50_0 = (self.m50!= -1)
        self.mask100_0 = (self.m100!= -1)
        self.mask150_0 = (self.m150 != -1)
        self.mask0_0 = self.mask0_0.type(torch.cuda.FloatTensor)

        self.mask50_0 = self.mask50_0.type(torch.cuda.FloatTensor)
        self.mask100_0 = self.mask100_0.type(torch.cuda.FloatTensor)
        self.mask150_0 = self.mask150_0.type(torch.cuda.FloatTensor)
        self.image_paths = input['A_paths' if AtoB else 'B20_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B50,self.fake_B100, self.fake_B150 = self.netG(self.real_A)  # G(A)


    def backward_D50(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B50), 1) 
        pred_fake50 = self.netD50(fake_AB.detach())
        self.loss_D_fake50 = self.criterionGAN(pred_fake50, False)

        real_AB = torch.cat((self.real_A, self.real_B50), 1)
        pred_real50 = self.netD50(real_AB)
        self.loss_D_real50 = self.criterionGAN(pred_real50, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake50 + self.loss_D_real50) * 0.5
        self.loss_D.backward()



    def backward_D100(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B100), 1) 
        pred_fake100 = self.netD100(fake_AB.detach())
        self.loss_D_fake100 = self.criterionGAN(pred_fake100, False)

        real_AB = torch.cat((self.real_A, self.real_B100), 1)
        pred_real100 = self.netD100(real_AB)
        self.loss_D_real100 = self.criterionGAN(pred_real100, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake100 + self.loss_D_real100) * 0.5
        self.loss_D.backward()

    def backward_D150(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B150), 1) 
        pred_fake150 = self.netD150(fake_AB.detach())
        self.loss_D_fake150 = self.criterionGAN(pred_fake150, False)

        real_AB = torch.cat((self.real_A, self.real_B150), 1)
        pred_real150 = self.netD150(real_AB)
        self.loss_D_real150 = self.criterionGAN(pred_real150, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake150 + self.loss_D_real150) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        mask50 = (self.real_B50 != -1)
        self.a50 = torch.masked_select(self.real_B50, mask50)
        self.b50 = torch.masked_select(self.fake_B50, mask50)

        mask100 = (self.real_B100 != -1)
        self.a100 = torch.masked_select(self.real_B100, mask100)
        self.b100 = torch.masked_select(self.fake_B100, mask100)



        mask150 = (self.real_B150 != -1)
        self.a150 = torch.masked_select(self.real_B150, mask150)
        self.b150 = torch.masked_select(self.fake_B150, mask150)

 
        fake_AB_50 = torch.cat((self.real_A, self.fake_B50), 1)
        pred_fake50 = self.netD50(fake_AB_50)
        self.loss_G_GAN_50 = self.criterionGAN(pred_fake50, True)
        self.loss_argo_50 = self.criterionL1(self.a50, self.b50) 

  
        fake_AB_100 = torch.cat((self.real_A, self.fake_B100), 1)
        pred_fake100 = self.netD100(fake_AB_100)
        self.loss_G_GAN_100 = self.criterionGAN(pred_fake100, True)
        self.loss_argo_100 = self.criterionL1(self.a100, self.b100)   

        fake_AB_150 = torch.cat((self.real_A, self.fake_B150), 1)
        pred_fake150 = self.netD150(fake_AB_150)
        self.loss_G_GAN_150 = self.criterionGAN(pred_fake150, True)
        self.loss_argo_150 = self.criterionL1(self.a150, self.b150) 

        x=torch.zeros(2,1,128,128)
        y=torch.zeros(2,1,128,128)


        self.fake_b100_0 = self.fake_B100*(self.mask50_0)
        self.fake_b50_1  = self.fake_B50*(self.mask50_0)

        self.fake_b150_0 = self.fake_B150*(self.mask100_0)
        self.fake_b100_1 = self.fake_B100*(self.mask100_0)


        self.loss_50_100 = self.criterionL1(torch.max(self.fake_b100_0-self.fake_b50_1+0.1,y.cuda(),out=None),x.cuda())
        self.loss_100_150 = self.criterionL1(torch.max(self.fake_b150_0-self.fake_b100_1+0.1,y.cuda(),out=None),x.cuda())


    def set_loss_G(self):
        self.backward_G()
        self.loss_G = torch.mean((100.0*self.loss_argo_50+self.loss_G_GAN_50) + (100.0*self.loss_argo_100+self.loss_G_GAN_100) + (100.0*self.loss_argo_150+self.loss_G_GAN_150)  +self.loss_50_100+self.loss_100_150)
        self.loss_G.backward()





    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.set_requires_grad([self.netD50,self.netD100,self.netD150], False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.set_loss_G()
        self.optimizer_G.step()             # udpate G's weights

