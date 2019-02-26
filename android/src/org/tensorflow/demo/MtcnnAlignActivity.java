package org.tensorflow.demo;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.bumptech.glide.request.RequestOptions;
import org.tensorflow.demo.env.Logger;

import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.MtcnnDetector;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class MtcnnAlignActivity extends Activity{
    private static final Logger LOGGER = new Logger();

    private Button pickpic;
    private ImageView show_image;
    private TextView show_text;
    private Classifier detector;
    private static final String MTCNN_MODEL_FILE = "file:///android_asset/mtcnn.pb";
    private float textSizePx;
    private float TEXT_SIZE_DIP = 18;
    private Context context;
    private Bitmap bitm;
    private Button start_rank;

    private final float confidence = 0.5f;
    private final float[] MTCNN_THRESHOLD = {0.4f, 0.7f, 0.8f};
    private final int MTCNN_FACE_NUMS = 20;
    private final int MTCNN_CROP_SIZE = 400;

    private float[] point1 = new float[2];
    private float[] point2 = new float[2];
    private float[] point3 = new float[2];
    private float[] point4 = new float[2];
    private float[] point5 = new float[2];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Set up the UI.
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mtcnn);

        pickpic = (Button)findViewById(R.id.photo_mtcnn);
        pickpic.setTextColor(Color.parseColor("#0D0068"));
        pickpic.setTextSize(20);
        pickpic.getBackground().setAlpha(0);
        pickpic.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        show_text.setText("");
                        pick_picture(v);
                    }
                }
        );

        start_rank = (Button)findViewById(R.id.align_mtcnn);
        start_rank.setTextColor(Color.parseColor("#0D0068"));
        start_rank.setTextSize(20);
        start_rank.getBackground().setAlpha(0);
        start_rank.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        detect(bitm);
                    }
                }
        );

        detector = MtcnnDetector.create(getAssets(), MTCNN_MODEL_FILE, MTCNN_THRESHOLD, MTCNN_FACE_NUMS);
        show_image = (ImageView)findViewById(R.id.show_image_mtcnn);
        show_text = (TextView)findViewById(R.id.result_text_mtcnn);
        context = getBaseContext();
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());

        request_permissions();
    }
    public void pick_picture(View view)
    {
        PhotoUtil.use_photo(MtcnnAlignActivity.this, 1001);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        String image_path;
        RequestOptions options = new RequestOptions().skipMemoryCache(true).diskCacheStrategy(DiskCacheStrategy.NONE);
        if (resultCode == Activity.RESULT_OK) {
            switch (requestCode) {
                case 1001:
                    if (data == null) {
                        return;
                    }

                    Uri image_uri = data.getData();

                    // Glide.with(RankActivity.this).load(image_uri).apply(options).into(show_image);
                    // get image path from uri
                    image_path = PhotoUtil.get_path_from_URI(MtcnnAlignActivity.this, image_uri);

                    get_bitmap(image_path);
                    break;

            }
        }
    }

    public void get_bitmap(String file_path)
    {
        Bitmap bitmap = BitmapFactory.decodeFile(file_path);
        bitm = bitmap;
        show_image.setImageBitmap(bitmap);
    }

    public void detect(Bitmap bitmap)
    {
        if (bitmap == null)
        {
            return;
        }
       /* float ratio = 1.0f;
        if (Math.min(bitmap.getWidth(), bitmap.getHeight())>500)
        {
            ratio = 500.0f / Math.min(bitmap.getWidth(), bitmap.getHeight());
        }
        else if (Math.max(bitmap.getWidth(), bitmap.getHeight())<200)
        {
            ratio = 200.0f / Math.max(bitmap.getWidth(), bitmap.getHeight());
        }
        int dstwidth = (int)(bitmap.getWidth() * ratio);
        int dstheight = (int)(bitmap.getHeight() * ratio);*/
        int dstwidth = MTCNN_CROP_SIZE;
        int dstheight = MTCNN_CROP_SIZE;
        Matrix frame_to_crop = ImageUtils.getTransformationMatrix(bitmap.getWidth(), bitmap.getHeight(),
                dstwidth,dstheight,0,true);
        Matrix crop_to_frame = new Matrix();
        frame_to_crop.invert(crop_to_frame);
        Bitmap bmResized = Bitmap.createBitmap(dstwidth,dstheight,Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmResized);
        canvas.drawBitmap(bitmap, frame_to_crop,null);

        Bitmap final_bitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas ori_canvas = new Canvas(final_bitmap);
        Matrix m = new Matrix();
        ori_canvas.drawBitmap(bitmap, m, null);

        final List<Classifier.Recognition> results = detector.recognizeImage(bmResized);
        String str = "检测到"+results.size() + "个人脸";
        show_text.setTextColor(Color.YELLOW);
        show_text.setText(str);

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            point1 = result.getPoint1();
            point2 = result.getPoint2();
            point3 = result.getPoint3();
            point4 = result.getPoint4();
            point5 = result.getPoint5();
            String st = ""+point1[0] + " " +point1[1];
            LOGGER.e(st);
            if (location != null)
            {
                crop_to_frame.mapRect(location);
                crop_to_frame.mapPoints(point1);
                crop_to_frame.mapPoints(point2);
                crop_to_frame.mapPoints(point3);
                crop_to_frame.mapPoints(point4);
                crop_to_frame.mapPoints(point5);
                String st1 = ""+point1[0] + " " +point1[1];
                LOGGER.e(st1);

                final Paint paint = new Paint();
                paint.setColor(Color.parseColor("#AA33AA"));
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(5.0f);
                ori_canvas.drawRect(location, paint);
                ori_canvas.drawCircle(point1[0], point1[1], 2, paint);
                ori_canvas.drawCircle(point2[0], point2[1], 2, paint);
                ori_canvas.drawCircle(point3[0], point3[1], 2, paint);
                ori_canvas.drawCircle(point4[0], point4[1], 2, paint);
                ori_canvas.drawCircle(point5[0], point5[1], 2, paint);

            }
        }
        show_image.setImageBitmap(final_bitmap);

    }
    private void request_permissions() {

        List<String> permissionList = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.CAMERA);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }

        // if list is not empty will request permissions
        if (!permissionList.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissionList.toArray(new String[permissionList.size()]), 1);
        }
    }

}

