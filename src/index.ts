import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

export {version} from './version';

export interface DetectedHelmet {
  detection_boxes: Int32Array,
  detection_classes: Int32Array
}


export async function load(path : string) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }

  const helmetDetection = new HelmetDetection(path);
  await helmetDetection.load();
  return helmetDetection;
}

export class HelmetDetection {
  private modelPath: string;
  private model: tfconv.GraphModel;

  constructor(path : string) {
    this.modelPath = path;
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

    const detection_boxes = result[2].dataSync() as Int32Array;
    const detection_classes = result[4].dataSync() as Int32Array;

    // clean the webgl tensors
    batched.dispose();
    tf.dispose(result);

    const objects: DetectedHelmet[] = [];

    objects.push({
      detection_boxes: detection_boxes,
      detection_classes: detection_classes
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
