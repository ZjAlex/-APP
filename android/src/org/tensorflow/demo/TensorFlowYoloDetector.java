/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Trace;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.env.SplitTimer;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class TensorFlowYoloDetector implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 23;

  private static final int NUM_CLASSES = 1;

  private static final int NUM_BOXES_PER_BLOCK = 3;

  private static final float con = 0.5f;

  // TODO(andrewharp): allow loading anchors and classes
  // from files.
  private static final double[] ANCHORS = {
    92, 122,
    144, 192,
    294, 388
  };
  private static final double[] ANCHORS_2 = {
          33, 43,
          45, 57,
          62, 82
  };
  private static final double[] ANCHORS_3 = {
          9, 12,
          17, 22,
          25, 32
  };




  private static final String[] LABELS = {
    "face"
  };

  // Config values.
  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intValues;
  private float[] floatValues;
  private String[] outputNames;
  private int image_width;
  private int image_height;

  private int blockSize;

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  /** Initializes a native TensorFlow session for classifying images. */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final int inputSize,
      final String inputName,
      final String outputName,
      final int blockSize,
      final int image_width,
      final int image_height) {
    TensorFlowYoloDetector d = new TensorFlowYoloDetector();
    d.inputName = inputName;
    d.inputSize = inputSize;

    // Pre-allocate buffers.
    d.outputNames = outputName.split(",");
    d.intValues = new int[inputSize * inputSize];
    d.floatValues = new float[inputSize * inputSize * 3];
    d.blockSize = blockSize;
    d.image_width = image_width;
    d.image_height = image_height;

    d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    return d;
  }

  private TensorFlowYoloDetector() {}

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  private float iou(RectF a, RectF b)
  {
    float a_x1 = a.left;
    float a_y1 = a.top;
    float a_x2 = a.right;
    float a_y2 = a.bottom;

    float b_x1 = b.left;
    float b_y1 = b.top;
    float b_x2 = b.right;
    float b_y2 = b.bottom;

    float min_x = Math.max(a_x1, b_x1);
    float min_y = Math.max(a_y1, b_y1);
    float max_x = Math.min(a_x2, b_x2);
    float max_y = Math.min(a_y2, b_y2);
    float w = Math.max(max_x - min_x, 0);
    float h = Math.max(max_y - min_y, 0);
    float interarea =  w * h;

    float a_w = Math.max(a_x2 - a_x1, 0);
    float a_h = Math.max(a_y2 - a_y1, 0);
    float b_w = Math.max(b_x2 - b_x1, 0);
    float b_h = Math.max(b_y2 - b_y1, 0);

    float area1 = a_w * a_h;
    float area2 = b_w * b_h;
    float eplison = 0.000001f;
    float iou = interarea / (area1 + area2 - interarea + eplison);
    return iou;
  }

  private RectF correct_box_size(float x, float y, float w, float h)
  {
    float ratio = Math.min(224.0f / image_width, 224.0f / image_height);
    int new_w = Math.round(image_width * ratio);
    int new_h = Math.round(image_height * ratio);

    float offset_w = (224 - new_w) / 2.0f / 224.0f;
    float offset_h = (224 - new_h) / 2.0f / 224.0f;
    float scale_w = 224.0f / new_w;
    float scale_h = 224.0f /new_h;
    float box_x = (x - offset_w) * scale_w;
    float box_y = (y - offset_h) * scale_h;
    float box_w = w * scale_w;
    float box_h = h * scale_h;
    box_x = box_x * image_width;
    box_y = box_y * image_height;
    box_w = box_w * image_width;
    box_h = box_h * image_height;

    RectF box = new RectF(box_x, box_y, box_w, box_h);
    return box;
  }

  private ArrayList<Recognition> nms(ArrayList<Recognition> boxes, int max_nums, float iou_thresh)
  {
    boxes.sort( new Comparator<Recognition>() {
      @Override
      public int compare(final Recognition lhs, final Recognition rhs) {
        // Intentionally reversed to put high confidence at the head of the queue.
        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
      }
    });
    ArrayList<Recognition> keep = new ArrayList<Recognition>();
    int curr_nums = 0;
    while (!boxes.isEmpty())
    {
      Recognition reg = boxes.get(0);
      RectF loc = reg.getLocation();
      keep.add(reg);
      curr_nums += 1;
      if (curr_nums >=max_nums)
      {
        break;
      }

      for(int i = 1; i < boxes.size(); i++)
      {
        RectF loc1 = boxes.get(i).getLocation();
        float iou_val = iou(loc, loc1);
        if (iou_val > iou_thresh)
        {
          boxes.remove(i);
          i--;
        }
      }
      boxes.remove(0);
    }
    return keep;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    final SplitTimer timer = new SplitTimer("recognizeImage");

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    for (int i = 0; i < intValues.length; ++i) {
      floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
      floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
      floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
    Trace.endSection();

    timer.endSplit("ready for inference");

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    timer.endSplit("ran inference");

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    //int gridWidth = bitmap.getWidth() / blockSize;
    //int gridHeight = bitmap.getHeight() / blockSize;
    int gridWidth = 224 / blockSize;
    int gridHeight = 224 / blockSize;
    final float[] output =
        new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
    final float[] output_2 =
            new float[(gridWidth*2) * (gridHeight*2) * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
    final float[] output_3 =
            new float[(gridWidth*4) * (gridHeight*4) * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
    inferenceInterface.fetch(outputNames[0], output);
    inferenceInterface.fetch(outputNames[1], output_2);
    inferenceInterface.fetch(outputNames[2], output_3);
    Trace.endSection();

    // Find the best detections.
    final PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            1,
            new Comparator<Recognition>() {
              @Override
              public int compare(final Recognition lhs, final Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    ArrayList<Recognition> boxes_array = new ArrayList<Recognition>();
    // output_1
    for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset =
              (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                  + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                  + (NUM_CLASSES + 5) * b;

          final float xPos = (x + expit(output[offset + 0])) * blockSize;
          final float yPos = (y + expit(output[offset + 1])) * blockSize;

          final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]);// * blockSize;
          final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]);// * blockSize;

         // RectF box = correct_box_size(xPos, yPos, w, h);

          final RectF rect =
              new RectF(
                  Math.max(0, xPos - w / 2),
                  Math.max(0, yPos - h / 2),
                  Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                  Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(output[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > con) {
            LOGGER.i(
                "%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect);
            pq.add(new Recognition("1_" + offset, LABELS[detectedClass], confidenceInClass, rect));
            boxes_array.add(new Recognition("1_" + offset, LABELS[detectedClass], confidenceInClass, rect));
          }
        }
      }
    }

    // output_2
    gridHeight = gridHeight * 2;
    gridWidth = gridWidth * 2;
    for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset =
                  (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                          + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                          + (NUM_CLASSES + 5) * b;

          final float xPos = (x + expit(output_2[offset + 0])) * blockSize / 2;
          final float yPos = (y + expit(output_2[offset + 1])) * blockSize / 2;

          final float w = (float) (Math.exp(output_2[offset + 2]) * ANCHORS_2[2 * b + 0]);// * blockSize;
          final float h = (float) (Math.exp(output_2[offset + 3]) * ANCHORS_2[2 * b + 1]);// * blockSize;

          final RectF rect =
                  new RectF(
                          Math.max(0, xPos - w / 2),
                          Math.max(0, yPos - h / 2),
                          Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                          Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(output_2[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output_2[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > con) {
            LOGGER.i(
                    "%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect);
            pq.add(new Recognition("2_" + offset, LABELS[detectedClass], confidenceInClass, rect));
            boxes_array.add(new Recognition("2_" + offset, LABELS[detectedClass], confidenceInClass, rect));
          }
        }
      }
    }

    // output_3
    gridHeight = gridHeight * 2;
    gridWidth = gridWidth * 2;
    for (int y = 0; y < gridHeight; ++y) {
      for (int x = 0; x < gridWidth; ++x) {
        for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
          final int offset =
                  (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                          + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                          + (NUM_CLASSES + 5) * b;

          final float xPos = (x + expit(output_3[offset + 0])) * blockSize / 4;
          final float yPos = (y + expit(output_3[offset + 1])) * blockSize / 4;

          final float w = (float) (Math.exp(output_3[offset + 2]) * ANCHORS_3[2 * b + 0]);// * blockSize;
          final float h = (float) (Math.exp(output_3[offset + 3]) * ANCHORS_3[2 * b + 1]);// * blockSize;

          final RectF rect =
                  new RectF(
                          Math.max(0, xPos - w / 2),
                          Math.max(0, yPos - h / 2),
                          Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                          Math.min(bitmap.getHeight() - 1, yPos + h / 2));
          final float confidence = expit(output_3[offset + 4]);

          int detectedClass = -1;
          float maxClass = 0;

          final float[] classes = new float[NUM_CLASSES];
          for (int c = 0; c < NUM_CLASSES; ++c) {
            classes[c] = output_3[offset + 5 + c];
          }
          softmax(classes);

          for (int c = 0; c < NUM_CLASSES; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > con) {
            LOGGER.i(
                    "%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect);
            pq.add(new Recognition("3_" + offset, LABELS[detectedClass], confidenceInClass, rect));
            boxes_array.add(new Recognition("3_" + offset, LABELS[detectedClass], confidenceInClass, rect));
          }
        }
      }
    }
    timer.endSplit("decoded results");

    ArrayList<Recognition> boxes_after_nms = new ArrayList<Recognition>();
    boxes_after_nms = nms(boxes_array, MAX_RESULTS, 0.f);

   /* final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
      recognitions.add(pq.poll());
    }*/
    Trace.endSection(); // "recognizeImage"

    timer.endSplit("processed results");

    return boxes_after_nms;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
