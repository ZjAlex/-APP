package org.tensorflow.demo;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
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
import android.widget.EditText;
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
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class FaceClassifierActivity extends Activity{
    private static final Logger LOGGER = new Logger();

    private Button pickpic;
    private Button cls;
    private Button get_face_feats;
    private Button save_feat;
    private Button clear_features;
    private EditText label;

    private ImageView show_image;
    private ImageView show_image1;
    private TextView show_text;
    private Classifier detector;
    private static final String MTCNN_MODEL_FILE = "file:///android_asset/mtcnn.pb";
    private float textSizePx;
    private float TEXT_SIZE_DIP = 18;
    private Context context;
    private Bitmap bitm;
    private Button extract_feat;

    private Map<String, float[]> map = new HashMap<String, float[]>();

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
        setContentView(R.layout.activity_face_cls);

        squeezenet = SqueezeNetDetector.create(getAssets(), SQUEEZENET_MODEL_FILE);

        pickpic = (Button)findViewById(R.id.photo_face_cls);
        pickpic.setTextColor(Color.parseColor("#0D0068"));
        pickpic.setTextSize(20);
        pickpic.getBackground().setAlpha(0);
        pickpic.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        show_text.setText("");
                        label.setText("");
                        show_image1.setImageBitmap(null);
                        pick_picture(v);
                    }
                }
        );

        extract_feat = (Button)findViewById(R.id.extrat_face_cls);
        extract_feat.setTextColor(Color.parseColor("#0D0068"));
        extract_feat.setTextSize(20);
        extract_feat.getBackground().setAlpha(0);
        extract_feat.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        extract_features();
                    }
                }
        );

        cls = (Button)findViewById(R.id.cls_face_cls);
        cls.setTextColor(Color.parseColor("#0D0068"));
        cls.setTextSize(20);
        cls.getBackground().setAlpha(0);
        cls.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        cls();
                    }
                }
        );

        get_face_feats = (Button)findViewById(R.id.get_face_cls);
        get_face_feats.setTextColor(Color.parseColor("#0D0068"));
        get_face_feats.setTextSize(20);
        get_face_feats.getBackground().setAlpha(0);
        get_face_feats.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        get_feats();
                    }
                }
        );

        save_feat = (Button)findViewById(R.id.save_face_cls);
        save_feat.setTextColor(Color.parseColor("#0D0068"));
        save_feat.setTextSize(20);
        save_feat.getBackground().setAlpha(0);
        save_feat.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        save();
                    }
                }
        );

        clear_features = (Button)findViewById(R.id.clear_face_cls);
        clear_features.setTextColor(Color.parseColor("#0D0068"));
        clear_features.setTextSize(20);
        clear_features.getBackground().setAlpha(0);
        clear_features.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        clear_feats();
                    }
                }
        );


        detector = MtcnnDetector.create(getAssets(), MTCNN_MODEL_FILE, MTCNN_THRESHOLD, MTCNN_FACE_NUMS);
        ((MtcnnDetector) detector).setFactor(MTCNN_FACTOR);
        ((MtcnnDetector) detector).setMinsize(MTCNN_MIN_SIZE);
        show_image = (ImageView)findViewById(R.id.show_image_face1_cls);
        show_image1 = (ImageView)findViewById(R.id.show_image_face2_cls);

        show_text = (TextView)findViewById(R.id.result_text_face_cls);
        show_text.setTextSize(20);
        show_text.setTextColor(Color.YELLOW);

        label = (EditText)findViewById(R.id.edit_text_cls);
        label.setTextSize(20);
        label.setTextColor(Color.YELLOW);

        context = getBaseContext();
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());

        request_permissions();
    }
    public void pick_picture(View view)
    {
        PhotoUtil.use_photo(FaceClassifierActivity.this, 1001);
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
                    image_path = PhotoUtil.get_path_from_URI(FaceClassifierActivity.this, image_uri);

                    get_bitmap(image_path);
                    break;

            }
        }
    }

    public void get_bitmap(String file_path)
    {
        Bitmap bitmap = BitmapFactory.decodeFile(file_path);
        bitm = bitmap;
        show_image.setImageBitmap(bitm);
    }

    public void extract_features()
    {
        String lab = label.getText().toString();
        if (lab.equals(""))
        {
            Toast toast = Toast.makeText(context, "请输入名字", Toast.LENGTH_SHORT);
            toast.show();
            return;
        }
        Bitmap bitmap = bitm;
        if (bitmap == null)
        {
            Toast toast = Toast.makeText(context, "没有图片", Toast.LENGTH_SHORT);
            toast.show();
            return;
        }
        Bitmap face1 = get_face_area(bitmap);
        if (face1 == null)
        {
            Toast toast = Toast.makeText(context, "没有检测到脸", Toast.LENGTH_SHORT);
            toast.show();
            return;
        }
        float[] face_feat1 = get_face_feature(face1);

        map.put(lab, face_feat1);
        Toast toast = Toast.makeText(context, "特征获取完毕", Toast.LENGTH_SHORT);
        toast.show();
    }

    private void cls()
    {
        Bitmap bitmap = bitm;
        if (bitmap == null)
        {
            Toast toast = Toast.makeText(context, "没有图片", Toast.LENGTH_SHORT);
            toast.show();
            return;
        }
        Bitmap face1 = get_face_area(bitmap);
        if (face1 == null)
        {

            Toast toast = Toast.makeText(context, "没有检测到人脸", Toast.LENGTH_SHORT);
            toast.show();
            return;
        }
        float[] face_feat1 = get_face_feature(face1);
        float min = 10000.0f;
        String l = "";
        for (Map.Entry<String, float[]> entry : map.entrySet())
        {
            float dis = 0.0f;
            for (int j = 0; j < 128; ++j)
            {
                dis += ((face_feat1[j] - entry.getValue()[j]) * (face_feat1[j] - entry.getValue()[j]));
            }
            dis = (float) (Math.sqrt(dis));
            if (dis < min)
            {
                min = dis;
                l = entry.getKey();
            }

        }
        show_text.setText("识别结果：   " + l);
    }

    private void clear_feats()
    {
        map.clear();
        SharedPreferences sp = context.getSharedPreferences("data1", context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sp.edit();
        Map<String, ?> spAll = sp.getAll();
        int num = 0;
        for (String key : spAll.keySet())
        {
            editor.remove(key);
            num += 1;
        }
        editor.commit();
        Toast toast = Toast.makeText(context,"成功清除特征: " + num, Toast.LENGTH_SHORT);
        toast.show();
    }

    private void get_feats()
    {
        map.clear();
        SharedPreferences sp = context.getSharedPreferences("data1", context.MODE_PRIVATE);

        Map<String, ?> spAll = sp.getAll();
        int num = 0;

        for (String key : spAll.keySet())
        {
            String values = sp.getString(key, null);
            if (values == null) break;
            num += 1;
            String[] str = values.split("#");
            float[] my_feats = new float[128];
            for (int j = 0; j < 128; ++j)
            {
                my_feats[j] = Float.parseFloat(str[j]);
            }
            map.put(key, my_feats);

        }
        Toast toast = Toast.makeText(context,"成功加载特征: " + num, Toast.LENGTH_SHORT);
        toast.show();
    }

    private void save()
    {
        if (map.size() == 0)
        {
            Toast toast = Toast.makeText(context,"特征数为0", Toast.LENGTH_SHORT);
            toast.show();
            return;
        }
        SharedPreferences sp = context.getSharedPreferences("data1", context.MODE_PRIVATE);
        SharedPreferences.Editor edit = sp.edit();

        for (Map.Entry<String, float[]> entry : map.entrySet())
        {
            String str_save = "";
            StringBuilder sb = new StringBuilder(str_save);
            for (int j = 0; j < 128; ++j)
            {
                sb.append(entry.getValue()[j]);
                if (j != 127)
                {
                    sb.append("#");
                }
            }
            edit.putString(entry.getKey(), sb.toString());
        }

        edit.commit();
        Toast toast = Toast.makeText(context,"成功存储特征: " + map.size(), Toast.LENGTH_SHORT);
        toast.show();
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
            Toast toast = Toast.makeText(context, "多于一张人脸", Toast.LENGTH_SHORT);
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

                show_image1.setImageBitmap(face);
                return face;
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

