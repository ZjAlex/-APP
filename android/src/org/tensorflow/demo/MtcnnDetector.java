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
public class MtcnnDetector implements Classifier {
    private static final Logger LOGGER = new Logger();

    // Only return this many results with at least this confidence.
    private static final float[] MINSIZE = {20.0f};

    private static final float[] FACTOR = {0.709f};

    private float[] THRESHOLDS = new float[3];
    private int face_nums;

    // Config values.
    private String inputName1 = "input";
    private String inputName2 = "min_size";
    private String inputName3 = "thresholds";
    private String inputName4 = "factor";

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private String[] outputNames = {"prob", "landmarks", "box"};

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /** Initializes a native TensorFlow session for classifying images. */
    public static MtcnnDetector create(
            final AssetManager assetManager,
            final String modelFilename,
            final float[] threshold,
            int face_nums_) {
        MtcnnDetector d = new MtcnnDetector();
        d.THRESHOLDS = threshold;
        d.face_nums = face_nums_;
        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        return d;
    }

    public void setFactor(float factor)
    {
        FACTOR[0] = factor;
    }

    public void setMinsize(float min_size)
    {
        MINSIZE[0] = min_size;
    }

    private MtcnnDetector() {}

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        final SplitTimer timer = new SplitTimer("recognizeImage");

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        intValues = new int[bitmap.getWidth() * bitmap.getHeight()];
        floatValues = new float[bitmap.getHeight() * bitmap.getWidth() * 3];
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF);
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF);
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF);
        }
        Trace.endSection(); // preprocessBitmap
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName1, floatValues,  bitmap.getHeight(), bitmap.getWidth(), 3);
        inferenceInterface.feed(inputName2, MINSIZE);
        inferenceInterface.feed(inputName4, FACTOR);
        inferenceInterface.feed(inputName3, THRESHOLDS,3);
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

        float[] output = new float[face_nums * 10];
        float[] output_2 = new float[face_nums * 4];
        float[] output_3 = new float[face_nums];
        inferenceInterface.fetch(outputNames[1], output);
        inferenceInterface.fetch(outputNames[2], output_2);
        inferenceInterface.fetch(outputNames[0], output_3);
        Trace.endSection();

        timer.endSplit("decoded results");

        ArrayList<Recognition> boxes = new ArrayList<Recognition>();
        for (int i =0; i < face_nums; ++i)
        {
            if (output_3[i] > 0)
            {
                final RectF rect =
                        new RectF(
                                Math.max(0, (output_2[i * 4 + 1])),
                                Math.max(0, (output_2[i * 4 + 0])),
                                Math.min(bitmap.getWidth() - 1, (output_2[i * 4 + 3])),
                                Math.min(bitmap.getHeight() - 1, (output_2[i * 4 + 2])));
               Recognition r = new Recognition("MTCNN" + i, "face",output_3[i], rect);

                float[] point1 = new float[2];
                float[] point2 = new float[2];
                float[] point3 = new float[2];
                float[] point4 = new float[2];
                float[] point5 = new float[2];
               point1[0] = output[i * 10 + 5];
               point1[1] = output[i * 10];
               point2[0] = output[i * 10 + 6];
               point2[1] = output[i * 10 + 1];
               point3[0] = output[i * 10 + 7];
               point3[1] = output[i * 10 + 2];
               point4[0] = output[i * 10 + 8];
               point4[1] = output[i * 10 + 3];
               point5[0] = output[i * 10 + 9];
               point5[1] = output[i * 10 + 4];
               String st = ""+point1[0] + " " +point1[1];
               LOGGER.e(st);
               r.setPoint1(point1);
               r.setPoint2(point2);
               r.setPoint3(point3);
               r.setPoint4(point4);
               r.setPoint5(point5);

               boxes.add(r);
            }
        }


        Trace.endSection(); // "recognizeImage"

        timer.endSplit("processed results");



        return boxes;
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
