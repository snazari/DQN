
Virtual env configuration for DQN scripts:

-----------

sudo apt-get install virtualenv
sudo apt-get install python3-venv
mkdir projectFolder
cd projectFolder
python3 -m venv venv  # creation of venv folder inside

source venv/bin/activate  # get inside environment

pip install tensorflow
## also tested with tf 1.0.0 :
# pip install tensorflow==1.0.0
pip install gym[atari]
pip install scipy
pip install opencv-python

# now the env is ready to run DQN scripts, e g :
python gym_dqn_atari.py pretrained/Seaquest-DQN-10M.ckpt True

deactivate  # leave environment


#----
# in case of probblems installing of importing opencv, try to install it outside virtualenv:

sudo apt-get install libopencv-dev python-opencv  # 

-------------


The resulting output is similar to the following:

(venv) user@user-vb:~/repos/DQN$ python gym_dqn_atari.py pretrained/Seaquest-DQN-10M.ckpt True
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  return f(*args, **kwds)
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  return f(*args, **kwds)

Usage:
   gym_dqn_atari.py  [optional: path_to_ckpt_file] [optional: True/False test mode]


WARN: gym.spaces.Box autodetected dtype as <class 'numpy.uint8'>. Please provide explicit dtype.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# of parameters in network  qnet :  3300019   ->   3.3 M
# of parameters in network  target_qnet :  3300019   ->   3.3 M
Double DQN
WARNING:tensorflow:From gym_dqn_atari.py:86: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
WARNING:tensorflow:From gym_dqn_atari.py:89: all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Please use tf.global_variables instead.
Traceback (most recent call last):
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1022, in _do_call
    return fn(*args)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1004, in _run_fn
    status, run_metadata)
  File "/usr/lib/python3.5/contextlib.py", line 66, in __exit__
    next(self.gen)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/framework/errors_impl.py", line 469, in raise_exception_on_not_ok_status
    pywrap_tensorflow.TF_GetCode(status))
tensorflow.python.framework.errors_impl.NotFoundError: Tensor name "qnet/Variable_6/RMSProp" not found in checkpoint files pretrained/Seaquest-DQN-10M.ckpt
     [[Node: save/RestoreV2_31 = RestoreV2[dtypes=[DT_FLOAT], _device="/job:localhost/replica:0/task:0/cpu:0"](_recv_save/Const_0, save/RestoreV2_31/tensor_names, save/RestoreV2_31/shape_and_slices)]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "gym_dqn_atari.py", line 100, in <module>
    saver.restore(session, sys.argv[1])
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 1439, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 767, in run
    run_metadata_ptr)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 965, in _run
    feed_dict_string, options, run_metadata)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1015, in _do_run
    target_list, options, run_metadata)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1035, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: Tensor name "qnet/Variable_6/RMSProp" not found in checkpoint files pretrained/Seaquest-DQN-10M.ckpt
     [[Node: save/RestoreV2_31 = RestoreV2[dtypes=[DT_FLOAT], _device="/job:localhost/replica:0/task:0/cpu:0"](_recv_save/Const_0, save/RestoreV2_31/tensor_names, save/RestoreV2_31/shape_and_slices)]]

Caused by op 'save/RestoreV2_31', defined at:
  File "gym_dqn_atari.py", line 89, in <module>
    saver = tf.train.Saver(tf.all_variables())
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 1051, in __init__
    self.build()
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 1081, in build
    restore_sequentially=self._restore_sequentially)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 675, in build
    restore_sequentially, reshape)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 402, in _AddRestoreOps
    tensors = self.restore_op(filename_tensor, saveable, preferred_shard)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/training/saver.py", line 242, in restore_op
    [spec.tensor.dtype])[0])
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/ops/gen_io_ops.py", line 668, in restore_v2
    dtypes=dtypes, name=name)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 763, in apply_op
    op_def=op_def)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 2395, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/user/repos/DQN/venv/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1264, in __init__
    self._traceback = _extract_stack()

NotFoundError (see above for traceback): Tensor name "qnet/Variable_6/RMSProp" not found in checkpoint files pretrained/Seaquest-DQN-10M.ckpt
     [[Node: save/RestoreV2_31 = RestoreV2[dtypes=[DT_FLOAT], _device="/job:localhost/replica:0/task:0/cpu:0"](_recv_save/Const_0, save/RestoreV2_31/tensor_names, save/RestoreV2_31/shape_and_slices)]]
