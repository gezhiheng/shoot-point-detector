import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'
import { loadAllRaw, createDataset } from './prepare_data.js'
import { buildModel } from './model.js'

async function run() {
  // 使用 CPU 后端
  await tf.setBackend('cpu')

  // 1. 加载并预处理数据
  const raw = await loadAllRaw('data/raw', 'data/labels')
  const { xs, ys } = createDataset(raw, 30, 1)

  // 2. 划分训练和验证集
  const split = Math.floor(xs.shape[0] * 0.8)
  const xTrain = xs.slice([0, 0, 0], [split, -1, -1])
  const yTrain = ys.slice([0], [split])
  const xVal = xs.slice([split, 0, 0], [-1, -1, -1])
  const yVal = ys.slice([split], [-1])

  // 3. 构建与编译模型
  const model = buildModel(30, 2)
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  })

  // 4. 模型训练
  await model.fit(xTrain, yTrain, {
    validationData: [xVal, yVal],
    epochs: 20,
    batchSize: 32,
  })

  // 5. 保存模型到文件系统
  await model.save('file://./model_tfjs')
  console.log('✅ 训练完成，模型已保存到 model_tfjs/')
}

run().catch(err => console.error(err))
