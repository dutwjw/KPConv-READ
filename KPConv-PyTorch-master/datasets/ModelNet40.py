

     0=================================0
     |    kernel point convolutions    |
     0=================================0


----------------------------------------------------------------------------------------------------------------------

     class handling modelnet40 dataset.
     implements a dataset, a sampler, and a collate_fn

----------------------------------------------------------------------------------------------------------------------

     hugues thomas - 11/06/2018



----------------------------------------------------------------------------------------------------------------------

          imports and global variables
      \**********************************/


# common libs
import time
import numpy as np
import pickle
import torch
import math


# os functions
from os import listdir
from os.path import exists, join

# dataset parent class
from datasets.common import pointclouddataset
from torch.utils.data import sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors

# ----------------------------------------------------------------------------------------------------------------------
#
#           dataset class definition
#       \******************************/


class ModelNet40Dataset(PointCloudDataset):
    """class to handle modelnet 40 dataset."""

    def __init__(self, config, train=true, orient_correction=true):
        """
        this dataset is small enough to be stored in-memory, so load all point clouds here
        """
        pointclouddataset.__init__(self, 'modelnet40')#实例化子类的__init__，就不会调用父类的__init__

        ############
        # parameters
        ############

        # dict from labels to names
        self.label_to_names = {0: 'airplane',
                               1: 'bathtub',
                               2: 'bed',
                               3: 'bench',
                               4: 'bookshelf',
                               5: 'bottle',
                               6: 'bowl',
                               7: 'car',
                               8: 'chair',
                               9: 'cone',
                               10: 'cup',
                               11: 'curtain',
                               12: 'desk',
                               13: 'door',
                               14: 'dresser',
                               15: 'flower_pot',
                               16: 'glass_box',
                               17: 'guitar',
                               18: 'keyboard',
                               19: 'lamp',
                               20: 'laptop',
                               21: 'mantel',
                               22: 'monitor',
                               23: 'night_stand',
                               24: 'person',
                               25: 'piano',
                               26: 'plant',
                               27: 'radio',
                               28: 'range_hood',
                               29: 'sink',
                               30: 'sofa',
                               31: 'stairs',
                               32: 'stool',
                               33: 'table',
                               34: 'tent',
                               35: 'toilet',
                               36: 'tv_stand',
                               37: 'vase',
                               38: 'wardrobe',
                               39: 'xbox'}

        # initialize a bunch of variables concerning class labels
		# initialize all label parameters given the label_to_names dict
        self.init_labels()

        # list of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # dataset folder
        self.path = '../../data/modelnet40'

        # type of task conducted on this dataset
        self.dataset_task = 'classification'

        # update number of class and data task in configuration

        config.num_classes = self.num_classes
        # self.num_classes = len(self.label_to_names)  common.py
        config.dataset_task = self.dataset_task

        # parameters from config
        self.config = config

        # training or test set
        self.train = train

        # number of models and models used per epoch
        # batch size is 1
        if self.train:
            self.num_models = 9843  
            #config.batch_num = 10  config.epoch_steps = 1000  epoch_steps is the Number of steps per epochs
            if config.epoch_steps and config.epoch_steps * config.batch_num < self.num_models:
                self.epoch_n = config.epoch_steps * config.batch_num
            else:
                self.epoch_n = self.num_models
        else:
            self.num_models = 2468
            self.epoch_n = min(self.num_models, config.validation_size * config.batch_num)
        ## Number of validation examples per epoch : validation_size
        
        #############
        # load models
        #############

        if 0 < self.config.first_subsampling_dl <= 0.01:#降采样的尺寸，边长为first_subsampling_dl立方体之内的点被认为是一个点
            raise valueerror('subsampling_parameter too low (should be over 1 cm')

        self.input_points, self.input_normals, self.input_labels = self.load_subsampled_clouds(orient_correction)

        return

    def __len__(self):
        """
        return the length of data here
        """
        return self.num_models

    def __getitem__(self, idx_list):
        """
        the main thread gives a list of indices to load a batch. each worker is going to work in parallel to load a
        different list of indices.
        """

        ###################
        # gather batch data
        ###################

        tp_list = []
        tn_list = []
        tl_list = []
        ti_list = []
        s_list = []
        r_list = []

        for p_i in idx_list:#idx_list包含了很多文件的点  做多4,5个吧

            # get points and labels
            points = self.input_points[p_i].astype(np.float32)
            normals = self.input_normals[p_i].astype(np.float32)
            label = self.label_to_idx[self.input_labels[p_i]]

            # data augmentation
            points, normals, scale, r = self.augmentation_transform(points, normals)#common.py r是旋转矩阵 scale是缩放

            # stack batch
            tp_list += [points]
            tn_list += [normals]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [scale]
            r_list += [r]

        ###################
        # concatenate batch
        ###################

        #show_modelnet_examples(tp_list, cloud_normals=tn_list)

        stacked_points = np.concatenate(tp_list, axis=0)#链接完后，是（N1+N2+N3,3）
        stacked_normals = np.concatenate(tn_list, axis=0)
        labels = np.array(tl_list, dtype=np.int64)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)#储存了每个文件的点的数量
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(r_list, axis=0)

        # input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)#在modelnet40中，feature是一列1
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, stacked_normals))#水平拼接
        else:
            raise valueerror('only accepted input dimensions are 1, 4 and 7 (without and with xyz)')

        #######################
        # create network inputs
        #######################
        #
        #   points, neighbors, pooling indices for each layers
        #

        # get the whole input list
        input_list = self.classification_inputs(stacked_points,
                                                stacked_features,
                                                labels,
                                                stack_lengths)

        # add scale and rotation for testing
        input_list += [scales, rots, model_inds]

        return input_list#注意这个input_list 里面的数据 有很多

    def load_subsampled_clouds(self, orient_correction):

        # restart timer
        t0 = time.time()

        # load wanted points if possible
        if self.train:
            split ='training'
        else:
            split = 'test'

        print('\nloading {:s} points subsampled at {:.3f}'.format(split, self.config.first_subsampling_dl))
        filename = join(self.path, '{:s}_{:.3f}_record.pkl'.format(split, self.config.first_subsampling_dl)) #序列化文件

        if exists(filename):# determine whether the file exists
            with open(filename, 'rb') as file:
                input_points, input_normals, input_labels = pickle.load(file)

        # else compute them from original points
        else:

            # collect training file names
            if self.train:
                names = np.loadtxt(join(self.path, 'modelnet40_train.txt'), dtype=np.str)
            else:
                names = np.loadtxt(join(self.path, 'modelnet40_test.txt'), dtype=np.str)#包括了各个小文件

            # initialize containers
            input_points = []#降采样完之后的点
            input_normals = []

            # advanced display
            n = len(names)
            progress_n = 30#空格的大小
            fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'

            # collect point clouds
            for i, cloud_name in enumerate(names):

                # read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = join(self.path, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # subsample them
                if self.config.first_subsampling_dl > 0:
                    #data是x,y,z,feature是 x,y,z后面的
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampledl=self.config.first_subsampling_dl)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                print('', end='\r')
                print(fmt_str.format('#' * ((i * progress_n) // n), 100 * i / n), end='', flush=true)

                # add to list
                input_points += [points]#转化成列表输出
                input_normals += [normals]

            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), end='', flush=true)
            print()

            # get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            input_labels = np.array([self.name_to_label[name] for name in label_names])#所有文件对应的label

            # save for later use
            with open(filename, 'wb') as file:
                pickle.dump((input_points,
                             input_normals,
                             input_labels), file)#储存每一个文件的采样后的点，法向量，以及label

        lengths = [p.shape[0] for p in input_points]#储存了点的个数
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} mb loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))#计算文件大小

        if orient_correction:#是否调换一下 y,z坐标
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]
            input_normals = [nn[:, [0, 2, 1]] for nn in input_normals]

        return input_points, input_normals, input_labels

# ----------------------------------------------------------------------------------------------------------------------
#
#           utility classes definition
#       \********************************/
# 对每个采样器，都需要提供__iter__方法，这个方法用以表示数据遍历的方式和__len__方法，用以返回数据的长度

class ModelNet40Sampler(sampler):
    """sampler for modelnet40"""
	#dataset: modelnet40dataset 参数类型指定
    def __init__(self, dataset: modelnet40dataset, use_potential=true, balance_labels=false):
        sampler.__init__(self, dataset)

        # does the sampler use potential for regular sampling
        self.use_potential = use_potential

        # should be balance the classes when sampling
        self.balance_labels = balance_labels

        # dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = none

        # initialize value for batch limit (max number of points per batch).
        self.batch_limit = 10000

        return

    def __iter__(self):
        """
        yield next batch indices here
        """

        ##########################################
        # initialize the list of generated indices
        ##########################################

        if self.use_potential:
            if self.balance_labels:

                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1#epoch_n是文件的数量，model的数量 num_classes是类别的数量，pick_n是每个类别的平均数量
                for i, l in enumerate(self.dataset.label_values):#不同的label ModelNet40是 40个模型

                    # get the potentials of the objects of this class
					#input_label的维度 [epoch_n,1]
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]#返回input_labels中符合当前类别的坐标
                    class_potentials = self.potentials[label_inds]#这是一个随机生成的很小的数，注意数据的类型，这不是list，是numpy.ndarray，numpy.ndarray可以用numpy.ndarray作为下标索引提取元素，list不行

                    # get the indices to generate thanks to potentials
                    if pick_n < class_potentials.shape[0]:#如果该类别的model个数大于平局值（期望值）
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]#argpartition 的作用是把这个list的第pick_n大的数字找出来，比它小的数放到它前边，比它大的数放到它后边，但并不会进行排序，保持原顺序
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])#如果小于的话就全都要
                    class_indices = label_inds[pick_indices]#再次进行随机选取indice，注意数据的类型，这不是list，是numpy.ndarray，numpy.ndarray可以用numpy.ndarray作为下标索引提取元素，list不行
                    gen_indices.append(class_indices)#拼接list准备输出 append几次，gen_indices的长度就是多少

                # stack the chosen indices of all classes
                gen_indices = np.random.permutation(np.hstack(gen_indices))#np.hstack 水平拼接 np.array 和 list类型的变量都可以接受,然后进行随机排序，各种随机

            else:

                # get indices with the minimum potential
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[:self.dataset.epoch_n]
                else:
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)#展示了，如何随机打乱一个数组或者是list

            # update potentials (change the order for the next epoch)
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])#应该都是1吧
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1#平均每个类别的数量 ！！！！！！！！！！
                gen_indices = []
                for l in self.dataset.label_values:#把数提取出来，应该是 0-39
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]#np.where返回满足条件的索引。[0]表示提取对应的纵坐标
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=true)#随机抽取pick_n个数，并且可以重复
                    gen_indices += [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))#permutation 对行进行随机排序
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ################
        # generator loop
        ################
		#上面的操作相当于randomsampling
        # initialize concatenation lists
        ti_list = []
        batch_n = 0

        # generator loop
        for p_i in gen_indices:
            
            # size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]#点的数量 保证一个batch不超过 self.batch_limit 个数的点

            # in case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # add data to current batch
            ti_list += [p_i]

            # update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)
		#yield 的作用就是把一个函数变成一个 generator

        return 0

    def __len__(self):
        """
        the number of yielded samples is variable
        """
        return none

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=false):
        """
        method performing batch and neighbors calibration.
            batch calibration: set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        neighbors calibration: set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. there is a limit for each layer.
        """

        ##############################
        # previously saved calibration
        ##############################

        print('\nstarting calibration (use verbose=true for more details)')
        t0 = time.time()

        redo = false

        # batch limit
        # ***********

        # load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # check if the batch limit associated with current parameters exists
        key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                   self.dataset.config.batch_num)
        if key in batch_lim_dict:#判断这个key是不是在batch_lin_dict的键值中
            self.batch_limit = batch_lim_dict[key]
        else:
            redo = true

        if verbose:
            print('\nprevious calibration found:')
            print('check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.okgreen
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.fail
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.endc))

        # neighbors limit
        # ***************

        # load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):#对于modelnet40分类来说 num_layers = 5

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = true

        if verbose:
            print('check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)#2cm * 2**layer_ind
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius#deform_radius is 5
                else:
                    r = dl * self.dataset.config.conv_radius#conv_radius is 2.5
                key = '{:.3f}_{:.3f}'.format(dl, r)
                if key in neighb_lim_dict:
                    color = bcolors.okgreen
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.fail
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.endc))

        if redo:

            ############################
            # neighbors calib parameters
            ############################

            # from config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.conv_radius + 1) ** 3))

            # histogram of neighborhood sizes 列是小方块的个数，是一个球，一个小方块是0.02m 行是卷积网络的层数
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # batch calib parameters
            ########################

            # estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num#这是期望的每一个batch中文件的个数 这是10

            # calibration parameters
            low_pass_t = 10
            kp = 100.0
            finer = false

            # convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # loop parameters
            last_display = time.time()
            i = 0
            breaking = false

            #####################
            # perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):
                    #在 neighbors.cpp 中 不够max_count的用support.size填充，所以neighb_mat.numpy() < neighb_mat.shape[0] 是计算有效的点的个数
                    # update neighborhood histogram #这个neighbors 对应了 collate_fn函数中的neighbors
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]#只要hist_n之内的数据 用直方图统计了每个点领域点数量的分布情况
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_t

                    # estimate error (noisy)
                    error = target_b - b

                    # save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # update batch limit with p controller
                    self.batch_limit += kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_t = 100
                        finer = true

                    # convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = true
                        break

                    i += 1
                    t = time.time()

                    # console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.batch_limit)))

                if breaking:
                    break

            # use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.t, axis=0) #实现元素的累加
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)#统计每个layer应该有多少个voxel有点，一共180个voxel
            self.dataset.neighborhood_limits = percentiles

            if verbose:
                # crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]
                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.fail
                        else:
                            color = bcolors.okgreen
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.endc)
                    print(line0)
                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # save batch_limit dictionary
            key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                       self.dataset.config.batch_num)
            batch_lim_dict[key] = self.batch_limit
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class modelnet40custombatch:
    """custom batch definition with memory pinning for modelnet40"""

    def __init__(self, input_list):#接受的是数据 待会儿分析
        #indices = next(self.sample_iter) 这个地方sample_iter返回一个list是文件索引，可能有4,5个
        #batch = self.collate_fn([self.dataset[i] for i in indices]) 这个地方也是一个列表  
        # get rid of batch dimension
        input_list = input_list[0] #这个地方应该结合dataloader看 input_list[0]正好对应第一个文件的所有input_list所包含的数据

        # number of layers
        l = (len(input_list) - 5) // 4 
        #这里减去5的意义 input_list += [scales, rots, model_inds]   li += [stacked_features, labels] 减去的是这几个维度
        #除以4的意义 li = input_points + input_neighbors + input_pools + input_stack_lengths 一共4个，得到的是“多层总数据的量”
        # extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+l]]#看懂了
        ind += l
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+l]]
        ind += l
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+l]]
        ind += l
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+l]]
        ind += l
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

        return self

    def unstack_points(self, layer=none):
        """unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=none):
        """unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=none):
        """unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=none, to_numpy=true):
        """
        return a list of the stacked elements in the batch at a certain layer. if no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise valueerror('unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is none or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def modelnet40collate(batch_data):
    return modelnet40custombatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           debug functions
#       \*********************/


def debug_sampling(dataset, sampler, loader):
    """shows which labels are sampled according to strategy chosen"""
    label_sum = np.zeros((dataset.num_classes), dtype=np.int32)
    for epoch in range(10):

        for batch_i, (points, normals, labels, indices, in_sizes) in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            label_sum += np.bincount(labels.numpy(), minlength=dataset.num_classes)
            print(label_sum)
            #print(sampler.potentials[:6])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=true)
    print(counts)


def debug_timing(dataset, sampler, loader):
    """timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # new time
            t = t[-1:]
            t += [time.time()]

            # update estim_b (low pass filter)
            estim_b += (len(batch.labels) - estim_b) / 100

            # pause simulating computations
            time.sleep(0.050)
            t += [time.time()]

            # average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b))

        print('************* epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=true)
    print(counts)


def debug_show_clouds(dataset, sampler, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        l = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # print characteristics of input tensors
            print('\npoints tensors')
            for i in range(l):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nneigbors tensors')
            for i in range(l):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\npools tensors')
            for i in range(l):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nstack lengths')
            for i in range(l):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nfeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nlabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\naugment scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\naugment rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nmodel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nare input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=true)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, sampler, loader):
    """timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # new time
            t = t[-1:]
            t += [time.time()]

            # pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'step {:08d} -> average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=true)
    print(counts)


class modelnet40workerinitdebug:
    """callable class that initializes workers."""

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __call__(self, worker_id):

        # print workers info
        worker_info = get_worker_info()
        print(worker_info)

        # get associated dataset
        dataset = worker_info.dataset  # the dataset copy in this worker process

        # in windows, each worker has its own copy of the dataset. in linux, this is shared in memory
        print(dataset.input_labels.__array_interface__['data'])
        print(worker_info.dataset.input_labels.__array_interface__['data'])
        print(self.dataset.input_labels.__array_interface__['data'])

        # configure the dataset to only process the split workload

        return
