from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt


class VisualizePredictionsHook(tf.train.SessionRunHook):
  """Saves graph of predictions overlaid on top of ground truth light curves."""

  def __init__(self, save_freq=10):
      '''Initializes the VisualizePredictionsHook class.

      Args:
        save_freq: int; how long to wait (in batches) between logging graphs.
      '''
      self.iteration = 0
      self.save_freq = save_freq

  def before_run(self, run_context):
      '''Sets Tensors to be retreived before call to run().

      Args:
        run_context: A SessionRunContext object.
      '''
      graph = tf.get_default_graph()
      loc = graph.get_tensor_by_name("loc:0")
      scale = graph.get_tensor_by_name("scale:0")
      target = graph.get_tensor_by_name("target:0")
      example_id = graph.get_tensor_by_name("example_id:0")

      return tf.train.SessionRunArgs([loc, scale, target, example_id])

  def after_run(self, run_context, run_values):
      '''Creates and saves a graph of the light curve and predictions.

      Args:
        run_context: A SessionRunContext object.
        run_values: A SessionRunValues object.
      '''
      if self.iteration % self.save_freq == 0:
          loc, scale, target, example_id = run_values.results
          x = np.arange(len(target[0]))

          plt.figure(figsize=(20, 10))
          plt.plot(x, target[0], label='Ground Truth')
          plt.errorbar(x, loc[0], fmt='.', alpha=0.3, yerr=scale[0],
                       label='Prediction')

          plt.title(example_id[0])
          plt.xlabel('Observation Number')
          plt.ylabel('Flux')
          plt.legend()

          path = os.path.join('visualizations', str(self.iteration))
          if not os.path.exists(path):
              os.makedirs(path)
          plt.savefig(path + '/' + str(example_id[0]) + '.pdf')
          plt.close()

      self.iteration += 1
