package org.tensorflow.demo;


import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class RankModel {
    private int input_size;
    private String[] output_names;
    private  String input_name;
    private TensorFlowInferenceInterface inferenceInterface;
    private float[] floatValues;
    private int[] intValues;
    private boolean logStats = false;
    public RankModel(
            AssetManager assetManager_,
            String input_name_,
            String[] output_names_,
            int input_size_ ,
            String model_path
    )
    {
        input_size = input_size_;
        output_names = output_names_;
        input_name = input_name_;
        inferenceInterface = new TensorFlowInferenceInterface(assetManager_, model_path);
        floatValues = new float[input_size * input_size * 3];
        intValues = new int[input_size * input_size];
    }

    public float rank(Bitmap bitmap)
    {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
        }
        inferenceInterface.feed(input_name, floatValues, 1, input_size, input_size, 3);
        inferenceInterface.run(output_names, logStats);
        float[] output = new float[11];
        inferenceInterface.fetch(output_names[0], output);
        float score = 0;
        int rank_level = 0;
        float max_score = 0;
        for (int i = 0; i < 11; ++i)
        {
            if (output[i] > max_score)
            {
                max_score = output[i];
                rank_level = i;
            }
            score += output[i]*i;
        }
        score *= 20;
        score = score % 100;
        return score;
    }
}
