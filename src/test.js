import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'
import { loadAllRaw, createDataset } from './prepare_data.js'

async function test() {
  await tf.setBackend('cpu')

  // 加载模型
  const model = await tf.loadLayersModel('file://./model_tfjs/model.json')

  // 加载测试数据
  const raw = await loadAllRaw('data/raw', 'data/labels')
  const { xs, ys } = createDataset(raw, 30, 1)
  const split = Math.floor(xs.shape[0] * 0.8)
  const xTest = xs.slice([split, 0, 0], [-1, -1, -1])
  const yTest = ys.slice([split], [-1])

  // 推理并计算准确率
  const preds = model.predict(xTest)
  const predIdx = preds.argMax(-1)
  const correct = predIdx.equal(yTest).sum().arraySync()
  const total = yTest.shape[0]
  const labels = await yTest.array()
  const positives = labels.filter(l => l === 1).length
  console.log(`测试样本总数: ${total}, 有击发点的窗口: ${positives}, 无击发点: ${total - positives}`)

  const allPreds = predIdx.arraySync()
  const counts = allPreds.reduce((acc, v) => { acc[v] = (acc[v]||0)+1; return acc }, {})
  console.log('模型预测的帧索引分布：', counts)
  console.log(`测试准确率: ${(correct/total*100).toFixed(2)}%`)  
}

test().catch(err => console.error(err))
