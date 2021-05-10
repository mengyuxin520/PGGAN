import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class Unaligned8Dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase8 + 'A')  # create a path '/path/to/data/trainA'
        #self.dir_B20 = os.path.join(opt.dataroot, opt.phase + 'B20')  # create a path '/path/to/data/trainB1'

        #self.dir_B2 = os.path.join(opt.dataroot, opt.phase + 'B2')  # create a path '/path/to/data/trainB2'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        #self.B20_paths = sorted(make_dataset(self.dir_B20, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        #self.B2_paths = sorted(make_dataset(self.dir_B2, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
       # self.B20_size = len(self.B20_paths)  # get the size of dataset B
        #self.B2_size = len(self.B2_paths)

        #self.dir_A50 = os.path.join(opt.dataroot, opt.phase + 'A50')  # create a path '/path/to/data/trainA'
        self.dir_B50 = os.path.join(opt.dataroot, opt.phase8 + 'B50')  # create a path '/path/to/data/trainB1'
        #self.dir_B2 = os.path.join(opt.dataroot, opt.phase + 'B2')  # create a path '/path/to/data/trainB2'
        #self.A50_paths = sorted(make_dataset(self.dir_A50, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B50_paths = sorted(make_dataset(self.dir_B50, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        #self.B2_paths = sorted(make_dataset(self.dir_B2, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        #self.A50_size = len(self.A50_paths)  # get the size of dataset A
        self.B50_size = len(self.B50_paths)  # get the size of dataset B
        #self.B2_size = len(self.B2_paths)

        self.dir_B100 = os.path.join(opt.dataroot, opt.phase8 + 'B100')  # create a path '/path/to/data/trainB1'
        self.B100_paths = sorted(
            make_dataset(self.dir_B100, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.B100_size = len(self.B100_paths)  # get the size of dataset B

        self.dir_B150 = os.path.join(opt.dataroot, opt.phase8 + 'B150')  # create a path '/path/to/data/trainB1'
        self.B150_paths = sorted(
            make_dataset(self.dir_B150, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.B150_size = len(self.B150_paths)  # get the size of dataset B



        self.dir_m0 = os.path.join(opt.dataroot, 'mask_0')  # create a path '/path/to/data/trainB1'
        self.m0_paths = sorted(
            make_dataset(self.dir_m0, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.m0_size = len(self.m0_paths)  # get the size of dataset B


        self.dir_m50 = os.path.join(opt.dataroot, 'mask_50')  # create a path '/path/to/data/trainB1'
        self.m50_paths = sorted(
            make_dataset(self.dir_m50, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.m50_size = len(self.m50_paths)  # get the size of dataset B



        self.dir_m100 = os.path.join(opt.dataroot, 'mask_100')  # create a path '/path/to/data/trainB1'
        self.m100_paths = sorted(
            make_dataset(self.dir_m100, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.m100_size = len(self.m100_paths)  # get the size of dataset B


        self.dir_m150 = os.path.join(opt.dataroot, 'mask_150')  # create a path '/path/to/data/trainB1'
        self.m150_paths = sorted(
            make_dataset(self.dir_m150, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.m150_size = len(self.m150_paths)  # get the size of dataset B


 

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        #if self.opt.serial_batches:   # make sure index is within then range
 

        A_img = Image.open(A_path).convert('L')
   
        A = self.transform_A(A_img)
      #  B20 = self.transform_B(B20_img)
        #B2 = self.transform_B(B2_img)


        index_B50 = index % self.B50_size
        B50_path = self.B50_paths[index_B50]
        B50_img = Image.open(B50_path).convert('L')
        B50 = self.transform_B(B50_img)



        index_B100 = index % self.B100_size
        B100_path = self.B100_paths[index_B100]
        B100_img = Image.open(B100_path).convert('L')
        B100 = self.transform_B(B100_img)

        index_B150 = index % self.B150_size
        B150_path = self.B150_paths[index_B150]
        B150_img = Image.open(B150_path).convert('L')
        B150 = self.transform_B(B150_img)


 

        index_m0 = 0
        m0_path = self.m0_paths[index_m0]
        m0_img = Image.open(m0_path).convert('L')
        m0 = self.transform_B(m0_img)
 
        index_m50 = 0
        m50_path = self.m50_paths[index_m50]
        m50_img = Image.open(m50_path).convert('L')
        m50 = self.transform_B(m50_img)

        index_m100 = 0
        m100_path = self.m100_paths[index_m100]
        m100_img = Image.open(m100_path).convert('L')
        m100 = self.transform_B(m100_img)

        index_m150 = 0
        m150_path = self.m150_paths[index_m150]
        m150_img = Image.open(m150_path).convert('L')
        m150 = self.transform_B(m150_img)



        return {'A': A, 'B50': B50,'B100': B100, 'B150': B150,  'A_paths': A_path, 'B50_paths': B50_path,'B100_paths': B100_path, 'B150_paths': B150_path, 'm0':m0, 'm50':m50,'m100':m100, 'm150':m150}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B50_size,  self.B100_size, self.B150_size)
