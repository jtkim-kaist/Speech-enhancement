"""
Convert model.ckpt to model.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# create a session
sess = tf.Session()

# import best model
save_path = '/home/jtkim/github/SE_ref/Speech-enhancement/SE/'
ckpt_path = '/home/jtkim/github/SE_ref/Speech-enhancement/SE/logs/logs_2018-08-24-01-50-12'
ckpt_name = sorted(glob.glob(ckpt_path + '/*.meta'))[-1]

saver = tf.train.import_meta_graph(ckpt_name) # graph
saver.restore(sess, ckpt_name.split('.meta')[0]) # variables

# get graph definition
gd = sess.graph.as_graph_def()
aa = [print(n.name) for n in gd.node]
print(aa)
# fix batch norm nodes
for node in gd.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in xrange(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

# generate protobuf
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd,
                                                                'model_1/pred,model_1/labels,model_1/cost'.split(","))
tf.train.write_graph(converted_graph_def, save_path, 'model.pb', as_text=False)