/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

import android.graphics.Bitmap;
import android.graphics.RectF;
import java.util.List;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {
  /**
   * An immutable result returned by a Classifier describing what was recognized.
   */
  public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /**
     * Display name for the recognition.
     */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;
    /**
     * face rank level */
    private int rank;
    /**
     * MTCNN five points (eyes, nose, lips)*/
    private float[] point1 = new float[2];
    private float[] point2 = new float[2];
    private float[] point3 = new float[2];
    private float[] point4 = new float[2];
    private float[] point5 = new float[2];

    /**
     * face features 128 dim */
    private float[] face_feature = new float[128];

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
      this.rank = -1;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public int getRank()
    {
      return rank;
    }

    public void setRank(int r){
      this.rank = r;
    }

    public float[] getPoint1(){ return point1;}
    public float[] getPoint2(){ return point2;}
    public float[] getPoint3(){ return point3;}
    public float[] getPoint4(){ return point4;}
    public float[] getPoint5(){ return point5;}

    public void setLocation(RectF location) {
      this.location = location;
    }
    public void setPoint1(final float[] point) {
      this.point1 = point;
    }
    public void setPoint2(final float[] point) {
      this.point2 = point;
    }
    public void setPoint3(final float[] point) {
      this.point3 = point;
    }
    public void setPoint4(final float[] point) {
      this.point4 = point;
    }
    public void setPoint5(final float[] point) {
      this.point5 = point;
    }

    public float[] getFace_feature(){return face_feature;};
    public void setFace_feature(final float[] face_feature){this.face_feature=face_feature;};

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  List<Recognition> recognizeImage(Bitmap bitmap);

  void enableStatLogging(final boolean debug);

  String getStatString();

  void close();
}
