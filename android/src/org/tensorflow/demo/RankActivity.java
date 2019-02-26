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

import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class RankActivity extends Activity{

    private Button pickpic;
    private RankModel rankModel;
    private ImageView show_image;
    private TextView show_text;
    private Classifier detector;
    private static final String YOLO_MODEL_FILE = "file:///android_asset/mobel_yolo_raw.pb";
    private static final int YOLO_INPUT_SIZE = 224;
    private static final String YOLO_INPUT_NAME = "image_input_1";
    private static final String YOLO_OUTPUT_NAMES ="output_1,output_2,output_3";
    private static final int YOLO_BLOCK_SIZE = 32;
    private float textSizePx;
    private float TEXT_SIZE_DIP = 18;
    private Context context;
    private Bitmap bitm;
    private Button start_rank;
    private float confidence = 0.3f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Set up the UI.
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rank);

        pickpic = (Button)findViewById(R.id.pickpic);
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

        start_rank = (Button)findViewById(R.id.start_rank);
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

        detector = TensorFlowYoloDetector.create(
                getAssets(),
                YOLO_MODEL_FILE,
                YOLO_INPUT_SIZE,
                YOLO_INPUT_NAME,
                YOLO_OUTPUT_NAMES,
                YOLO_BLOCK_SIZE,
                224,
                224
                );
        rankModel = new RankModel(getAssets(), "input", new String[]{"output_1"},
                64, "file:///android_asset/rank_model_final.pb");
        show_image = (ImageView)findViewById(R.id.show_image);
        show_text = (TextView)findViewById(R.id.result_text);
        context = getBaseContext();
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());

        request_permissions();
    }
    public void pick_picture(View view)
    {
        PhotoUtil.use_photo(RankActivity.this, 1001);
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
                    image_path = PhotoUtil.get_path_from_URI(RankActivity.this, image_uri);

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
        Matrix frame_to_crop = ImageUtils.getTransformationMatrix(bitmap.getWidth(), bitmap.getHeight(),
                224,224,0,true);
        Matrix crop_to_frame = new Matrix();
        frame_to_crop.invert(crop_to_frame);
        Bitmap bmResized = Bitmap.createBitmap(224,224,Bitmap.Config.ARGB_8888);
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

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();
        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();

            if (location != null && result.getConfidence() >= confidence) {

                final Paint paint = new Paint();
                paint.setColor(Color.parseColor("#AA33AA"));
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(5.0f);
                crop_to_frame.mapRect(location);
                location.left = Math.max(0, location.left);
                location.top = Math.max(0, location.top);
                location.right = Math.min(final_bitmap.getWidth(), location.right);
                location.bottom = Math.min(final_bitmap.getHeight(), location.bottom);
                Bitmap face = Bitmap.createBitmap(final_bitmap, (int)location.left,
                        (int)location.top, (int)(location.right - location.left),
                        (int)(location.bottom - location.top));

                Bitmap face_resized = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
                Canvas face_rs = new Canvas(face_resized);
                Matrix face_m = ImageUtils.getTransformationMatrix(face.getWidth(), face.getHeight(),
                        64,64,0,true);
                face_rs.drawBitmap(face, face_m, null);
                float rank = rankModel.rank(face_resized);
                ori_canvas.drawRect(location, paint);
                String label = "score: "+(int)rank;
                final float cornerSize = Math.min(location.width(), location.height()) / 6.0f;
                BorderedText borderedText;
                borderedText = new BorderedText(cornerSize);
                borderedText.setExteriorColor(Color.YELLOW);
                borderedText.setExteriorColor(Color.GREEN);
                borderedText.drawText(ori_canvas, location.left + cornerSize, location.bottom, label);
                result.setLocation(location);
                mappedRecognitions.add(result);
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
