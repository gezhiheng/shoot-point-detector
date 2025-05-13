import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'

/**
 * 构建 1D-CNN 模型
 */
export function buildModel(windowSize, featureDim = 2) {
  const model = tf.sequential()
  model.add(tf.layers.inputLayer({ inputShape: [windowSize, featureDim] }))
  model.add(tf.layers.conv1d({ filters: 16, kernelSize: 3, activation: 'relu' }))
  model.add(tf.layers.maxPool1d({ poolSize: 2 }))
  model.add(tf.layers.conv1d({ filters: 32, kernelSize: 3, activation: 'relu' }))
  model.add(tf.layers.globalAveragePooling1d())
  model.add(tf.layers.dense({ units: windowSize, activation: 'softmax' }))
  return model
}
