/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

const BASE_PATH = '/assets/json';

export {version} from './version';

export interface DetectedHelmet {
  num_detections: Float32Array,
  raw_detection_scores: Float32Array,
  detection_boxes: Float32Array,
  detection_scores: Float32Array,
  detection_classes: Float32Array,
  raw_detection_boxes: Float32Array,
  detection_multiclass_scores: Float32Array
}


export async function load() {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }

  const helmetDetection = new HelmetDetection();
  await helmetDetection.load();
  return helmetDetection;
}

export class HelmetDetection {
  private modelPath: string;
  private model: tfconv.GraphModel;

  constructor() {
    this.modelPath = BASE_PATH;
  }

  async load() {
    this.model = await tfconv.loadGraphModel(this.modelPath);

    // Warmup the model.
    const result = await this.model.executeAsync(tf.zeros([1, 300, 300, 3])) as
        tf.Tensor[];
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
  }

  /**
   * Infers through the model.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   */
  private async infer(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement): Promise<DetectedHelmet[]> {
    const batched = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
      }
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0);
    });

    // model returns two tensors:
    // 1. box classification score with shape of [1, 1917, 90]
    // 2. box location with shape of [1, 1917, 1, 4]
    // where 1917 is the number of box detectors, 90 is the number of classes.
    // and 4 is the four coordinates of the box.
    const result = await this.model.executeAsync(batched) as tf.Tensor[];

    const num_detections =  result[0].dataSync() as Float32Array;
    const raw_detection_scores = result[0].dataSync() as Float32Array;
    const detection_boxes = result[0].dataSync() as Float32Array;
    const detection_scores = result[0].dataSync() as Float32Array;
    const detection_classes = result[0].dataSync() as Float32Array;
    const raw_detection_boxes = result[0].dataSync() as Float32Array;
    const detection_multiclass_scores = result[0].dataSync() as Float32Array;

    // clean the webgl tensors
    batched.dispose();
    tf.dispose(result);

    const objects: DetectedHelmet[] = [];

    objects.push({
      num_detections: num_detections,
      raw_detection_scores: raw_detection_scores,
      detection_boxes: detection_boxes,
      detection_scores: detection_scores,
      detection_classes: detection_classes,
      raw_detection_boxes: raw_detection_boxes,
      detection_multiclass_scores: detection_multiclass_scores
    });

    return objects;
  }

  /**
   * Detect objects for an image returning a list of bounding boxes with
   * assocated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   *
   */
  async detect(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement): Promise<DetectedHelmet[]> {
    return this.infer(img);
  }

  /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
