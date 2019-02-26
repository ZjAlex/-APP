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
public class SqueezeNetDetector implements Classifier {
    private static final Logger LOGGER = new Logger();

    // Config values.
    private String inputName1 = "input";

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private String[] outputNames = {"output"};

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /** Initializes a native TensorFlow session for classifying images. */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename) {
        SqueezeNetDetector d = new SqueezeNetDetector();
        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
        return d;
    }

    private SqueezeNetDetector() {}

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
            floatValues[i * 3 + 0] = (((intValues[i] >> 16) & 0xFF) - 127.5f) / 128.0f;
            floatValues[i * 3 + 1] = (((intValues[i] >> 8) & 0xFF) - 127.5f) / 128.0f;
            floatValues[i * 3 + 2] = ((intValues[i] & 0xFF) - 127.5f) / 128.0f;
        }
        Trace.endSection(); // preprocessBitmap
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName1, floatValues,  1, bitmap.getHeight(), bitmap.getWidth(), 3);
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

        float[] output = new float[128];
        inferenceInterface.fetch(outputNames[0], output);
        Trace.endSection();

        timer.endSplit("decoded results");

        ArrayList<Recognition> boxes = new ArrayList<Recognition>();

        Recognition r = new Recognition("face", "face_feature", 0.0f, new RectF());

        r.setFace_feature(output);
        boxes.add(r);

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
