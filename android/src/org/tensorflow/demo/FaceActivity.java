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
import android.widget.Toast;

import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.bumptech.glide.request.RequestOptions;
import org.tensorflow.demo.env.Logger;

import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.MtcnnDetector;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class FaceActivity extends Activity{
    private static final Logger LOGGER = new Logger();

    private Button pickpic;
    private Button empty;
    private ImageView show_image;
    private ImageView show_image1;
    private ImageView show_image2;
    private ImageView show_image3;
    private TextView show_text;
    private Classifier detector;
    private static final String MTCNN_MODEL_FILE = "file:///android_asset/mtcnn.pb";
    private float textSizePx;
    private float TEXT_SIZE_DIP = 18;
    private Context context;
    private Bitmap bitm;
    private Bitmap bitm1;
    private Button start_rank;

    private final float[] MTCNN_THRESHOLD = {0.4f, 0.5f, 0.5f};
    private final int MTCNN_FACE_NUMS = 5;
    private final int MTCNN_CROP_SIZE = 400;
    private final float MTCNN_MIN_SIZE = 40.f;
    private final float MTCNN_FACTOR = 0.850f;

    private float[] point1 = new float[2];
    private float[] point2 = new float[2];
    private float[] point3 = new float[2];
    private float[] point4 = new float[2];
    private float[] point5 = new float[2];

    private int PhotoCounter = 0;
    private static final String SQUEEZENET_MODEL_FILE = "file:///android_asset/squeezenet.pb";
    private Classifier squeezenet;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Set up the UI.
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_ver);

        squeezenet = SqueezeNetDetector.create(getAssets(), SQUEEZENET_MODEL_FILE);

        pickpic = (Button)findViewById(R.id.photo_face);
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

        start_rank = (Button)findViewById(R.id.align_face);
        start_rank.setTextColor(Color.parseColor("#0D0068"));
        start_rank.setTextSize(20);
        start_rank.getBackground().setAlpha(0);
        start_rank.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        detect();
                    }
                }
        );

        start_rank = (Button)findViewById(R.id.empty);
        start_rank.setTextColor(Color.parseColor("#0D0068"));
        start_rank.setTextSize(20);
        start_rank.getBackground().setAlpha(0);
        start_rank.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        empty_images();
                    }
                }
        );

        detector = MtcnnDetector.create(getAssets(), MTCNN_MODEL_FILE, MTCNN_THRESHOLD, MTCNN_FACE_NUMS);
        ((MtcnnDetector) detector).setFactor(MTCNN_FACTOR);
        ((MtcnnDetector) detector).setMinsize(MTCNN_MIN_SIZE);
        show_image = (ImageView)findViewById(R.id.show_image_face1);
        show_image1 = (ImageView)findViewById(R.id.show_image_face2);
        show_image2 = (ImageView)findViewById(R.id.show_image_face3);
        show_image3 = (ImageView)findViewById(R.id.show_image_face4);

        show_text = (TextView)findViewById(R.id.result_text_face);
        context = getBaseContext();
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());

        request_permissions();
    }
    public void pick_picture(View view)
    {
        PhotoUtil.use_photo(FaceActivity.this, 1001);
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
                    image_path = PhotoUtil.get_path_from_URI(FaceActivity.this, image_uri);

                    get_bitmap(image_path);
                    break;

            }
        }
    }

    public void get_bitmap(String file_path)
    {
        Bitmap bitmap = BitmapFactory.decodeFile(file_path);
        if (PhotoCounter == 0)
        {
            bitm = bitmap;
            show_image.setImageBitmap(bitm);
            PhotoCounter = 1;
        }
        else if (PhotoCounter == 1)
        {
            bitm1 = bitmap;
            show_image1.setImageBitmap(bitm1);
            PhotoCounter = 0;
        }
    }

    private void empty_images()
    {
        show_image.setImageBitmap(null);
        show_image1.setImageBitmap(null);
        show_image2.setImageBitmap(null);
        show_image3.setImageBitmap(null);
        bitm = null;
        bitm1 = null;
        PhotoCounter = 0;
    }

    public void detect()
    {
        Bitmap bitmap = bitm;
        Bitmap bitmap1 = bitm1;
        if (bitmap == null || bitmap1 == null)
        {
            Toast toast = Toast.makeText(context, "没有两张图片", Toast.LENGTH_LONG);
            toast.show();
            return;
        }
        Bitmap face1 = get_face_area(bitmap);
        Bitmap face2 = get_face_area(bitmap1);
        if (face1 == null || face2 == null)
        {
            return;
        }
        float[] face_feat1 = get_face_feature(face1);
        float[] face_feat2 = get_face_feature(face2);
        float sum = 0.0f;
        for (int i = 0; i < 128; ++i)
        {
            sum += (face_feat1[i] - face_feat2[i]) * (face_feat1[i] - face_feat2[i]);
        }
        float distance = (float)(Math.sqrt(sum));
        String dis = "  相似度： " + distance;
        show_text.setText(dis);
    }

    private float[] get_face_feature(Bitmap face)
    {
        Bitmap face_resized = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888);
        Canvas face_rs = new Canvas(face_resized);
        Matrix face_m = ImageUtils.getTransformationMatrix(face.getWidth(), face.getHeight(),
                160,160,0,true);
        face_rs.drawBitmap(face, face_m, null);
        List<Classifier.Recognition> boxes = squeezenet.recognizeImage(face_resized);
        float[] feature = boxes.get(0).getFace_feature();
        return feature;
    }

    private Bitmap get_face_area(Bitmap bitmap)
    {
        int dstwidth = MTCNN_CROP_SIZE;
        int dstheight = MTCNN_CROP_SIZE;
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        int min = Math.min(w, h);
        int max = Math.max(w, h);
        int dst = (int)(MTCNN_CROP_SIZE * (max / min));
        dstwidth = dst;
        dstheight = dst;
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
        if (results.size() > 1)
        {
            Toast toast = Toast.makeText(context, "多于一张人脸", Toast.LENGTH_LONG);
            toast.show();
            return null;
        }

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

                float x1 = Math.max(location.left - 22, 0);
                float y1 = Math.max(location.top - 22, 0);
                float x2 = Math.min(location.right + 22, final_bitmap.getWidth() - 1);
                float y2 = Math.min(location.bottom + 22, final_bitmap.getHeight() - 1);

                RectF loc = new RectF(x1, y1, x2, y2);

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

                Bitmap face = Bitmap.createBitmap(final_bitmap, (int)location.left,
                    (int)location.top, (int)(location.right - location.left),
                    (int)(location.bottom - location.top));
                ori_canvas.drawRect(loc, paint);
                ori_canvas.drawCircle(point1[0], point1[1], 2, paint);
                ori_canvas.drawCircle(point2[0], point2[1], 2, paint);
                ori_canvas.drawCircle(point3[0], point3[1], 2, paint);
                ori_canvas.drawCircle(point4[0], point4[1], 2, paint);
                ori_canvas.drawCircle(point5[0], point5[1], 2, paint);

                if (PhotoCounter == 0)
                {
                    show_image2.setImageBitmap(face);
                    PhotoCounter = 1;
                    return face;
                }
                else if(PhotoCounter == 1)
                {
                    show_image3.setImageBitmap(face);
                    PhotoCounter = 0;
                    return face;
                }

            }
        }
        return null;
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

