import tensorflow as tf
from tensorflow import keras
from keras.regularizers import Regularizer 
###############------------------- FOR AZRA ---------------------#############

from tensorflow.keras.models import model_from_yaml, model_from_json
from tensorflow import keras
# from functions import *
import pickle, shapely
import scipy

from scipy.ndimage import gaussian_filter
import scipy
from scipy import signal, ndimage
from sklearn.model_selection import train_test_split
import numpy as np
import math as mt
import tempfile
import tensorflow as tf
import pickle
from scipy import misc
import glob, csv
from tensorflow.keras import layers
from shapely.geometry import box, Polygon, Point, LinearRing
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, concatenate, Concatenate
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
from keras.regularizers import Regularizer 
# from keras_gcn import GraphConv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from sklearn import preprocessing
from numpy import linalg as LA
import pandas as pd
from tensorflow.keras.utils import plot_model
from numpy import matlib
from progressbar import ProgressBar
import tensorflow_model_optimization as tfmot
from matplotlib import cm
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.constraints import max_norm
# main = "Bharat_local_runs/"

def rew_new(x, y, obj_boun, present=False):
  rew = []
  if present:
    for i in range(len(x)):
      kk = Point(x[i],y[i])
      for ii in obj_boun:
        bb = box(ii[0], ii[1], ii[2], ii[3])
        if bb.contains(kk):
          rew.append(1)
        else:
          rew.append(0)
  else:
    rew = [0]*len(x)

  return np.asarray(rew)


def rew(x, y, theta, objs, obj_boun, env, env_boun, present=False):
  reward = []
  # obj = k2
  # obj_boun = k3
  for i in range(len(x)): 
    if present:
      if present:
        for pp in range(len(objs)):
          k2 = obj[pp]
          k3 = obj_boun[pp]
          if((max(k3)[0] >= x[i] >= max(k2)[0]) and (max(k2)[1] >= y[i] >= min(k2)[1]) and (90 < theta[i] < 270)):
            reward.append(1)
          elif((min(k3)[0] <= x[i] <= min(k2)[0]) and (max(k2)[1] >= y[i] >= min(k2)[1]) and (90>theta[i] or theta[i]>270)):
            reward.append(1)
          elif((min(k3)[1] <= y[i] <= min(k2)[1]) and (max(k2)[0] >= x[i] >= min(k2)[0]) and (180>theta[i]>0 )):
            reward.append(1)
          elif((max(k3)[1] >= y[i] >= max(k2)[1]) and (max(k2[0]) >= x[i] >= min(k2)[0]) and (360>theta[i]>180)):
            reward.append(1)

          elif((min(k3)[0] <= x[i] <= min(k2)[0]) and (max(k3)[1] >= y[i] >= max(k2)[1]) and (15>theta[i] or theta[i]>255)):
            reward.append(1)
          elif((min(k3)[0] <= x[i] <= min(k2)[0]) and (min(k3)[1] <= y[i] <= min(k2[1])) and (105>theta[i] or theta[i]>335)):
            reward.append(1)
          elif((max(k3)[0] >= x[i] >= max(k2)[0]) and (min(k3)[1] <= y[i] <= min(k2)[1]) and (195>theta[i]>75)):
            reward.append(1)
          elif((max(k3)[0] >= x[i] >= max(k2)[0]) and (max(k3)[1] >= y[i] >= max(k2)[1]) and (285>theta[i]>165)):
            reward.append(1)

      if len(reward) != i+1:
        if((max(k1_env[0]) >= x[i] >= max(k1)[0]) and (90>theta[i] or theta[i]>270)):
          reward.append(0)
        elif((min(k1_env[0]) <= x[i] <= min(k1)[0]) and (270>theta[i]>90)):
          reward.append(0)  
        elif((max(k1_env[1]) >= y[i] >= max(k1)[1]) and (180>theta[i]>0)):
          reward.append(0)
        elif((min(k1_env)[1] <= y[i] <= min(k1)[1]) and (360>theta[i]>180)):
          reward.append(0)
        elif len(reward) != i+1:
            reward.append(0)
      
    else:
      if((max(k1_env[0]) >= x[i] >= max(k1)[0]) and (90>theta[i] or theta[i]>270)):
        reward.append(0)
      elif((min(k1_env[0]) <= x[i] <= min(k1)[0]) and (270>theta[i]>90)):
        reward.append(0)  
      elif((max(k1_env[1]) >= y[i] >= max(k1)[1]) and (180>theta[i]>0)):
        reward.append(0)
      elif((min(k1_env)[1] <= y[i] <= min(k1)[1]) and (360>theta[i]>180)):
        reward.append(0)
      elif len(reward) != i+1:
          reward.append(0)
    
  reward = np.asarray(reward)
  return reward


#%% HD
def HD(s, t):
    with open(main + 'hd_som_wt2.pk1', 'rb') as k:
        wt2 = pickle.load(k)
    
    # phase1d = np.zeros((100, 1))
    PI2d = np.zeros((10, 10))
    k = PI2d.shape  
    trj_hd_resp = []

    for j in range(len(s)):
        if (j%10000 == 0):
            print(j)
        X1 = [mt.cos(mt.radians(t[0])), mt.sin(mt.radians(t[0]))]
        X2 = [mt.cos(mt.radians(t[j])), mt.sin(mt.radians(t[j]))]
        s1 = X2[0]*X1[1] - X1[0]*X2[1]
        s2 = X2[0]*X1[0] + X1[1]*X2[1]
        # print(s2)
        X = [s1, s2]
        y_p = repsom2dlinear(X, wt2)
        trj_hd_resp.append(y_p)
    print("HD response computed")
    return trj_hd_resp

#%%
# def PI(resp, s):
#     X, Y, theta = np.zeros((100,1)), np.ones((100,1)), [[0]*100] 
#     bf = 2*6*mt.pi
#     dt = np.divide(1, 100)
#     betaa, t, Xbg, Ybg = 50, 0, 1, 0
#     tarr = []
#     for ii in range(1,len(resp)):
#         if (ii%10000 == 0):
#             print(ii)
        

#         y_q = resp[ii]
#         inp1d = np.reshape(np.transpose(y_q),(100,1))
#         theta_dot = [(bf + betaa * s[ii] * k[0] * 10) for k in inp1d]
#         theta_dot[:] = [x*dt for x in theta_dot]
#         theta.append([i+j for i,j in zip(theta[ii-1], theta_dot)])

#     theta = np.transpose(np.asarray(theta))
#     # print(theta.shape)
#     Xarr = np.cos(theta)
#     PI1d = Xarr
#     return PI1d


def repsom2dlinear(x, wt):
    sz_wt = list(wt.shape)
    y = np.zeros((sz_wt[0], sz_wt[1]))
    if(sz_wt[2] != len(x)):
        print('Invalid input size in repsom2d()\n')
        return

    for i in range(sz_wt[0]):
        for j in range(sz_wt[1]):
            v = wt[i][j].reshape(sz_wt[2], 1)    
            # print(v)
            y[i][j] =  np.dot(x,v)
    return y


def unitvec(pos_corr):
    temp1 = np.subtract(pos_corr[1:, :], pos_corr[:-1, :])
    temp2 = np.sqrt((temp1*temp1).sum(axis=1))
    temp3 = temp1 / temp2.reshape(temp1.shape[0],1)
    return temp3


def relu(input):
    if input > 0:
	    return input
    else:
	    return 0


def test_train(dat, p):
  train_dat = np.asarray([dat[k] for k in range(len(dat)) if not k%p==0])
  test_dat = np.asarray([dat[k] for k in range(len(dat)) if k%p==0])
  return [train_dat, test_dat]


def mse(data, pred_data):
  mse_ = np.sum(np.square(data - pred_data))/len(data)
  return mse_


def seq_data(data, seq_len):
  temp1 = []

  for i in range(seq_len, data.shape[0]):
    temp3 = data[i-seq_len:i]
    temp1.append(temp3)
  temp1 = np.asarray(temp1)
  
  return temp1


class FF(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(FF, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.j_h = tf.keras.layers.Dense(self.units)
        self.j_x = tf.keras.layers.Dense(self.units)
        self.k_h = tf.keras.layers.Dense(self.units)
        self.k_x = tf.keras.layers.Dense(self.units)

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        return {'units': self.units}
        
    def call(self, inputs, states):
        #print("FF:", inputs, states)
        prev_output = states[0]
        j = tf.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = tf.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]

def firing_rate_map(firposgrid, ot, firr, title):
    res = 45
    #firr = list(firr[0])
    x = np.arange(-1, 1, 1/res)
    y = np.arange(-1, 1, 1/res)
    fx,fy = np.meshgrid(x, y)
    firingmap = np.zeros(fx.shape)
    #gridpoint_x = np.asarray(np.reshape(fx, np.size(fx), 1))
    #gridpoint_y = np.asarray(np.reshape(fx, np.size(fx), 1))
    #gridpoint = np.transpose([gridpoint_x, gridpoint_y])
    #roundinggridpoint = np.round(gridpoint)
    #firposround = np.round(firposgrid)
    firingvalue = ot[firr]
    for ii in range(len(firposgrid)):
        q1 = np.argmin(abs(firposgrid[ii,0] - fx[1,:]))
        q2 = np.argmin(abs(firposgrid[ii,1] - fx[1,:]))
        firingmap[q1,q2] = firingvalue[ii]
    firingmap = firingmap/max(np.max(firingmap),1)
    gaussian = matlab_style_gauss2D([10, 10], 1.5)
    spikes_smooth = scipy.signal.convolve2d(gaussian, firingmap) 
    rotated_img = ndimage.rotate(spikes_smooth, 1*270)
    #np.rot90([spikes_smooth], 2)
    # plt.imshow(rotated_img, origin= 'upper')
    # plt.title(title)
    # plt.colorbar()
    # ax=plt.gca()                            # get the axis
    # ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    # ax.set_xlim(ax.get_xlim()[::-1])        # invert the axis
    # ax.xaxis.tick_bottom()                     # and move the X-Axis    
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])

    return rotated_img
  
def matlab_style_gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class GraphLayer(keras.layers.Layer):

    def __init__(self,
                 step_num=1,
                 activation=None,
                 **kwargs):
        """Initialize the layer.
        :param step_num: Two nodes are considered as connected if they could be reached in `step_num` steps.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        self.supports_masking = True
        super(GraphLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'step_num': self.step_num,
            'activation': self.activation,
        }
        base_config = super(GraphLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_walked_edges(self, edges, step_num):
        """Get the connection graph within `step_num` steps
        :param edges: The graph in single step.
        :param step_num: Number of steps.
        :return: The new graph that has the same shape with `edges`.
        """
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(K.batch_dot(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return K.cast(K.greater(deeper, 0.0), K.floatx())

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self.activation(self._call(features, edges))
        return outputs

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphConv(GraphLayer):
    r"""Graph convolutional layer.
    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.
        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.W, self.b = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = int(input_shape[0][-1])
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]

    def _call(self, features, edges):
        proj = K.dot(features, self.W)
        if self.use_bias:
            proj += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        # aggr = proj/2
        aggr = tf.math.divide((K.sum(proj, axis=1, keepdims=True) + K.epsilon()), 3)
        # aggr = K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), proj) \
        #     / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())
        return features + aggr
        # return features


def obj_cent(traj_nam):
  q1, q2, q3, q4 = [(0.4, 0.4), (-0.4, 0.4), (-0.4, -0.4), (0.4, -0.4)]
  obj_seq = [q3, q1, q2, q4]
  var_chk = traj_nam[11] 
  if var_chk == "1":
    obj_c = [q1]
    obj_c_plot = obj_seq[:-2]
  elif var_chk == "2":
    obj_c = [q2]
    obj_c_plot = obj_seq[:-1]
  elif var_chk == "3":
    obj_c = [q3]
    obj_c_plot = [obj_seq[0]]
  elif var_chk == "4":
    obj_c = [q4]
    obj_c_plot = obj_seq
  else:
    obj_c = [(0.0, 0.0)]
    obj_c_plot = [(0.4, 0.4), (-0.4, 0.4), (-0.4, -0.4), (0.4, -0.4)]

  return [obj_c, obj_c_plot]


def computer_fisher(model, imgset, img_ind):
  with tf.GradientTape(persistent=True) as gt:
      f_accum = []
      for i in range(len(model.weights)):
          f_accum.append(np.zeros(model.weights[i].shape))
      f_accum = np.array(f_accum)
      if len(img_ind):
        img_ind = img_ind
      else:
        img_ind = np.random.randint(imgset[0].shape[0], size = 100)
      # pbar = ProgressBar(maxval=num_sample).start()
      for j in range(len(img_ind)):
          # img_index = j
          # img_index = np.random.randint(imgset[0].shape[0])
          img_index = img_ind[j]
          for m in range(len(model.weights)):
              input_mod = [np.expand_dims(imgset[0][img_index],0), np.expand_dims(imgset[1][img_index],0)]
              # print(input_mod)
              model_output = model(input_mod)
              grads = gt.gradient(tf.math.log(model_output), model.weights)[m]
              f_accum[m] += np.square(grads[0])
          if j%100 == 0:
            print(j)
          # pbar.update(j+1)
      # pbar.finish()
      f_accum /= len(img_ind)
  del gt
  return f_accum


class ewc_reg(Regularizer):
    def __init__(self, fisher, prior_weights, Lambda):
        self.fisher = fisher
        self.prior_weights = prior_weights
        self.Lambda = Lambda

    def __call__(self, x):
        regularization = 0.
        regularization += self.Lambda * K.sum(self.fisher * K.square(x - self.prior_weights))
        return regularization

    def get_config(self):
        return {'Lambda': float(self.Lambda)}


def custom_lr_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    t_loss = mse(y_true, y_pred)

    if y_true == 1:
      return 1000*t_loss
    if y_true == 0:
      return t_loss

def PI(x, y, theta_rad, beta, pi_use, n=100, ):
  print(pi_use)
  if pi_use == "no_osc":
    #Calculating distance from starting point
    print("------using PI WITHOUT oscillators-----")
    pos = np.column_stack((x,y))
    a = pos[0,0] * np.ones(pos[:,0].shape)
    b = pos[0,1] * np.ones(pos[:,1].shape)
    origin = np.transpose(np.append([a],[b],axis=0)) #for different x,y

    disp = pos - origin

    # Head direction parameters
    dth = np.divide(2*np.pi, n)
    theta_pref = np.arange(0, 2*np.pi, dth)
    pref_dir = np.transpose([np.cos(theta_pref), np.sin(theta_pref)])
    print(len(pos))
    hdi = preprocessing.normalize(np.cos(np.matlib.repmat(theta_pref, len(pos),1) - np.transpose((np.matlib.repmat(theta_rad[0:len(pos)],n, 1)))), norm='l2')

    # HD responses
    hd_resp = []
    for i in range(len(disp)):
        for j in range(len(pref_dir)):
            z = np.array(disp[i])
            dj = np.array(pref_dir[j])
            hd_resp.append(np.dot(z,dj))
    hd_resp = np.transpose(np.reshape(hd_resp, (len(disp),len(pref_dir))))

    # path integraion
    pi_layer_beta = [] 
    for i in range(len(beta)):
        pi_layer_temp = np.sin(beta[i] * hd_resp)
        pi_layer_beta.append((pi_layer_temp))
    pi_layer_beta = np.asarray(pi_layer_beta)
    pi_beta = pi_layer_beta[0]
    for i in range(len(beta) - 1):
        pi_beta = np.concatenate((pi_beta, pi_layer_beta[i+1]))
    pi_lay = pi_beta.T


  ##### ---------------------------- PI (WITH OSCILLATORS) ----------------------#########
  if pi_use == "osc":
    # print("-------- using PI WITH oscillators --------")
    trj_hd_resp = HD(speed, theta)
    PI1d = PI(trj_hd_resp, speed)
    PI1d = np.transpose(PI1d)
    PI1d = preprocessing.normalize(PI1d, norm='l2', axis=1)

    hd_resp = [iii.T.reshape(100,1) for iii in trj_hd_resp]
    hd_resp = np.asarray(hd_resp).reshape(len(hd_resp), hd_resp[0].shape[0])
    hd_resp = preprocessing.normalize(hd_resp, norm='l2', axis=1)
    num_images = PI1d.shape[0]

    pi_lay = PI1d

  return pi_lay



def model_arch(data, nodes, activ, lr):
  print("#########-----------------TRAINING MODEL---------------#########")
  act = 'relu'
  input_img = Input(shape = data[0].shape[1:], name="IMAGE")
  input_pi = Input(shape = data[1].shape[1], name="PI")

  encoder = Conv2D(8, (5, 5), padding='same', activation= act, name="CONV_1")(input_img)
  encoder = MaxPooling2D(pool_size=(2,2), padding='same', name="MAXPOOL_1")(encoder)
  encoder = Conv2D(4, (5, 5), padding='same', activation= act,name="CONV_2")(encoder)
  encoder = MaxPooling2D(pool_size=(2,2), padding='same', name="MAXPOOL_2")(encoder)
  encoder = Conv2D(2, (5, 5), padding='same', activation= act,name="CONV_3")(encoder)
  encoder = MaxPooling2D(pool_size=(2,2), padding='same', name="MAXPOOL_3")(encoder)
  flatencoder=Flatten()(encoder) 
  #flatencoder = GlobalAveragePooling2D()(encoder)
  dense0 = Dense(nodes, activation = 'sigmoid', name='LEC')(flatencoder)
  dense_pi1 = Dense(nodes, activation= 'sigmoid', name='MEC')(input_pi)
  dense_pi1 = layers.Reshape((1,nodes))(dense_pi1)
  dense0 = layers.Reshape((1,nodes))(dense0)
  data_layer = layers.concatenate([dense0, dense_pi1], axis=1)
  print(data_layer.shape)
  edge_layer = tf.constant(np.matlib.repmat(np.asarray([[1/3,1/3], [1/3,1/3]]), 1, 1).reshape((1,2,2)))
  conv_layer = GraphConv(units=nodes, step_num=1)([data_layer, edge_layer])
  conv_layer0 = Flatten()(conv_layer) 
  dense1 = Dense(nodes, activation = activ, name = 'D1')(conv_layer0)
  dense2 = Dense(nodes, activation = activ, name = 'D2')(dense1)
  dense3 = Dense(nodes, activation = activ, name = 'D3')(dense2)

  output1 = Dense(1, activation='linear', name='VALUE1')(dense3)
  # output2 = Dense(1, activation='linear', name='VALUE2')(conv_layer[:,1,:])
  #
  regressor_model = Model([input_img, input_pi], output1)
  #regressor_model = Model([input_img,], [outputx, outputy, outputz])
  opt = tf.keras.optimizers.Adam(learning_rate= lr)
  regressor_model.compile(optimizer=opt, loss="mse", )
  # regressor_model.summary()

  return regressor_model



def model_arch_catas(data, old_mod_dict, nodes, lam, activ, lr, ):
  tf.keras.backend.clear_session()

  old_wts = old_mod_dict["weights"]
  I = old_mod_dict["fisher"]

  print("#########-----------------RETRAINING MODEL---------------#########")
  act = 'relu'
  input_img = Input(shape = data[0].shape[1:], name="IMAGE")
  input_pi = Input(shape = data[1].shape[1], name="PI")

  encoder = Conv2D(8, (5, 5), padding='same', activation= act, name="CONV_1", kernel_regularizer=ewc_reg(I[0], old_wts[0], Lambda = lam),
                   bias_regularizer = ewc_reg(I[1], old_wts[1], Lambda = lam))(input_img)
  encoder = MaxPooling2D(pool_size=(2,2), padding='same', name="MAXPOOL_1")(encoder)
  encoder = Conv2D(4, (5, 5), padding='same', activation= act, name="CONV_2", kernel_regularizer=ewc_reg(I[2], old_wts[2], Lambda = lam),
                   bias_regularizer = ewc_reg(I[3], old_wts[3], Lambda = lam))(encoder)
  encoder = MaxPooling2D(pool_size=(2,2), padding='same', name="MAXPOOL_2")(encoder)
  encoder = Conv2D(2, (5, 5), padding='same', activation= act, name="CONV_3", kernel_regularizer=ewc_reg(I[4], old_wts[4], Lambda = lam),
                   bias_regularizer = ewc_reg(I[5], old_wts[5], Lambda = lam))(encoder)
  encoder = MaxPooling2D(pool_size=(2,2), padding='same', name="MAXPOOL_3")(encoder)
  flatencoder = Flatten()(encoder) 
  #flatencoder = GlobalAveragePooling2D()(encoder)
  dense0 = Dense(nodes, activation = 'sigmoid', name='LEC', kernel_regularizer=ewc_reg(I[6], old_wts[6], Lambda = lam),
                 bias_regularizer = ewc_reg(I[7], old_wts[7], Lambda = lam))(flatencoder)
  # dense1 = Dense(50, activation = act, name='LEC2')(dense0)
  # dense0 = Dropout(0.5)(dense0)
  dense_pi1 = Dense(nodes, activation= 'sigmoid', name='MEC', kernel_regularizer=ewc_reg(I[8], old_wts[8], Lambda = lam),
                    bias_regularizer = ewc_reg(I[9], old_wts[9], Lambda = lam))(input_pi)
  # dense_pi2 = Dense(50, activation= act, name='MEC2')(dense_pi1)
  # concat = layers.concatenate([dense1, dense_pi2])
  dense_pi1 = layers.Reshape((1,nodes))(dense_pi1)
  dense0 = layers.Reshape((1,nodes))(dense0)
  data_layer = layers.concatenate([dense0, dense_pi1], axis=1, name = "Concat")
  print(data_layer.shape)
  # edge_layer = layers.Input(shape=(None, None))
  edge_layer = tf.constant(np.matlib.repmat(np.asarray([[1/3,1/3], [1/3,1/3]]), 1, 1).reshape((1,2,2)))
  # edge_layer = tf.constant(np.asarray([[0,1], [1,0.]]).reshape((1,2,2)))
  conv_layer = GraphConv(units=nodes, step_num=1, kernel_regularizer=ewc_reg(I[10], old_wts[10], Lambda = lam),
                         bias_regularizer = ewc_reg(I[11], old_wts[11], Lambda = lam), name = "graph_conv")([data_layer, edge_layer])
  # conv_layer = Dropout(0.5)(conv_layer)
  # conv_layer0 = layers.Add()([conv_layer[:,0,:],conv_layer[:,1,:]]) 
  conv_layer0 = Flatten()(conv_layer) 
  dense1 = Dense(nodes, activation=activ, name = 'D1', kernel_regularizer=ewc_reg(I[12], old_wts[12], Lambda = lam),
                 bias_regularizer = ewc_reg(I[13], old_wts[13], Lambda = lam))(conv_layer0)
  dense2 = Dense(nodes, activation=activ, name = 'D2', kernel_regularizer=ewc_reg(I[14], old_wts[14], Lambda = lam),
                 bias_regularizer = ewc_reg(I[-5], old_wts[15], Lambda = lam))(dense1)
  dense3 = Dense(nodes, activation=activ, name = 'D3', kernel_regularizer=ewc_reg(I[16], old_wts[16], Lambda = lam),
                 bias_regularizer = ewc_reg(I[-3], old_wts[17], Lambda = lam))(dense2)
  # dense4 = Dense(50, activation=act, name = 'D4')(dense3)

  output1 = Dense(1, activation='linear', name='VALUE1', kernel_regularizer=ewc_reg(I[18], old_wts[19], Lambda = lam),
                  bias_regularizer = ewc_reg(I[19], old_wts[19], Lambda = lam))(dense3)
  # output2 = Dense(1, activation='linear', name='VALUE2')(conv_layer[:,1,:])
  #
  model_r = Model([input_img, input_pi], output1)
  opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
  model_r.compile(optimizer=opt, loss = custom_lr_loss, )
  # model_r.summary()

  return model_r