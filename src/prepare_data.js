// src/prepare_data.js
import fs from 'fs/promises'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'

/**
 * 批量加载 data/raw/ 下所有原始 JSON 序列，并根据 data/labels/ 下对应的标注文件赋值 label。
 * @param {string} rawDir - 原始数据目录
 * @param {string} labelDir - 标注文件目录
 * @returns {Promise<Array<{time:number,yaw:number,pitch:number,label:number}>>}
 */
export async function loadAllRaw(rawDir, labelDir) {
  const files = await fs.readdir(rawDir)
  const raws = []
  for (const file of files) {
    if (!file.endsWith('.json')) continue
    // 1. 读取原始数据
    const data = await fs.readFile(`${rawDir}/${file}`, 'utf8')
    const arr = JSON.parse(data)
    // 2. 读取标注文件，期望结构 { shootIndex: number }
    let shootIndex = arr.length - 1
    try {
      const labelData = await fs.readFile(`${labelDir}/${file}`, 'utf8')
      const { shootIndex: idx } = JSON.parse(labelData)
      if (typeof idx === 'number' && idx >= 0 && idx < arr.length) {
        shootIndex = idx
      }
    } catch (e) {
      // 如果没有对应标注，默认最后一帧
    }
    // 3. 合并
    for (let i = 0; i < arr.length; i++) {
      const [time, yaw, pitch] = arr[i]
      raws.push({ time, yaw, pitch, label: i === shootIndex ? 1 : 0 })
    }
  }
  return raws
}

/**
 * 切窗、标准化并生成特征与标签 Tensor
 * @param {Array<{time:number,yaw:number,pitch:number,label:number}>} raw
 * @param {number} windowSize
 * @param {number} stride
 */
export function createDataset(raw, windowSize = 30, stride = 1) {
  const allYaw   = raw.map(r => r.yaw)
  const allPitch = raw.map(r => r.pitch)
  const meanYaw  = tf.mean(allYaw).arraySync()
  const stdYaw   = tf.moments(allYaw).variance.sqrt().arraySync()
  const meanP    = tf.mean(allPitch).arraySync()
  const stdP     = tf.moments(allPitch).variance.sqrt().arraySync()

  const X = []
  const y = []
  for (let start = 0; start + windowSize <= raw.length; start += stride) {
    const slice = raw.slice(start, start + windowSize)
    const normed = slice.map(r => [
      (r.yaw   - meanYaw)  / stdYaw,
      (r.pitch - meanP)    / stdP
    ])
    const idx = slice.findIndex(r => r.label === 1)
    if (idx >= 0) {
      X.push(normed)
      y.push(idx)
    }
  }

  return {
    xs: tf.tensor3d(X),              // [N, windowSize, 2]
    ys: tf.tensor1d(y, 'float32')      // [N]
  }
}